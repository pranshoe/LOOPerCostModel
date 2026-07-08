"""The LOOPerCostModel network: an AST-based recursive + recurrent regressor.

The model mirrors each program's loop-nest hierarchy. Bottom-up:

1. an LSTM embeds each computation's variable-length affine-transformation sequence;
2. an LSTM embeds each computation's expression (post-order operation vectors);
3. a feed-forward network merges each computation's static features with those two
   embeddings into a per-computation embedding;
4. a recursive layer summarizes child loops and computations up the loop-nest tree;
5. an LSTM aggregates the per-nest root embeddings into a whole-program embedding;
6. a regression head maps it to a single predicted speedup.
"""

import torch
from torch import nn

from .constants import MAX_TAGS


def _xavier(weight):
    return nn.init.xavier_uniform_(weight)


class LOOPerCostModel(nn.Module):
    """Recursive-LSTM speedup regressor. See the module docstring for the layer roles.

    Args:
        input_size: width of the static per-computation feature slice fed to the
            computation-embedding MLP (paired with the transformation and expression
            embeddings).
        comp_embed_layer_sizes: hidden widths of the computation-embedding MLP.
        drops: dropout probabilities per layer.
        loops_tensor_size: width of the per-loop feature vector.
    """

    def __init__(
        self,
        input_size=846,
        comp_embed_layer_sizes=(600, 350, 200, 180),
        drops=(0.225, 0.225, 0.225, 0.225),
        output_size=1,
        lstm_embedding_size=100,
        expr_embed_size=100,
        loops_tensor_size=8,
        num_layers=1,
        bidirectional=True,
    ):
        super().__init__()
        comp_embed_layer_sizes = list(comp_embed_layer_sizes)
        drops = list(drops)
        embedding_size = comp_embed_layer_sizes[-1]

        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [embedding_size * 2 + loops_tensor_size] + comp_embed_layer_sizes[-2:]
        comp_embed_layer_sizes = [
            input_size + lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size
        ] + comp_embed_layer_sizes

        # Linear projection applied to each affine-transformation tag before the LSTM.
        self.encode_vectors = nn.Linear(MAX_TAGS, MAX_TAGS, bias=True)

        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        for i in range(len(comp_embed_layer_sizes) - 1):
            layer = nn.Linear(comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True)
            _xavier(layer.weight)
            self.comp_embedding_layers.append(layer)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))

        self.regression_layers = nn.ModuleList()
        self.regression_dropouts = nn.ModuleList()
        for i in range(len(regression_layer_sizes) - 1):
            layer = nn.Linear(regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True)
            _xavier(layer.weight)
            self.regression_layers.append(layer)
            self.regression_dropouts.append(nn.Dropout(drops[i]))

        # Feed-forward network combining loop features with child embeddings.
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts = nn.ModuleList()
        for i in range(len(concat_layer_sizes) - 1):
            layer = nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True)
            nn.init.zeros_(layer.weight)
            self.concat_layers.append(layer)
            self.concat_dropouts.append(nn.Dropout(drops[i]))

        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
        _xavier(self.predict.weight)

        self.ELU = nn.ELU()
        self.LeakyReLU = nn.LeakyReLU(0.01)

        # Learned embeddings representing "no computations" / "no child loops" at a level.
        self.no_comps_tensor = nn.Parameter(_xavier(torch.zeros(1, embedding_size)))
        self.no_nodes_tensor = nn.Parameter(_xavier(torch.zeros(1, embedding_size)))

        self.comps_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        self.nodes_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        self.roots_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        self.transformation_vectors_embed = nn.LSTM(
            MAX_TAGS, lstm_embedding_size, batch_first=True,
            bidirectional=bidirectional, num_layers=num_layers,
        )
        self.exprs_embed = nn.LSTM(11, expr_embed_size, batch_first=True)

    @property
    def device(self):
        return self.predict.weight.device

    def get_hidden_state(self, node, comps_embeddings, loops_tensor):
        """Recursively embed a loop-nest node from its children and its own features."""
        device = comps_embeddings.device
        nodes_list = [
            self.get_hidden_state(child, comps_embeddings, loops_tensor)
            for child in node["child_list"]
        ]
        if nodes_list:
            nodes_tensor = torch.cat(nodes_list, 1)
            _, (nodes_h_n, _) = self.nodes_lstm(nodes_tensor)
            nodes_h_n = nodes_h_n.permute(1, 0, 2)
        else:
            nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(
                comps_embeddings.shape[0], -1, -1
            )

        if node["has_comps"]:
            selected_comps = torch.index_select(
                comps_embeddings, 1, node["computations_indices"].to(device)
            )
            _, (comps_h_n, _) = self.comps_lstm(selected_comps)
            comps_h_n = comps_h_n.permute(1, 0, 2)
        else:
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(
                comps_embeddings.shape[0], -1, -1
            )

        selected_loop = torch.index_select(loops_tensor, 1, node["loop_index"].to(device))
        x = torch.cat((nodes_h_n, comps_h_n, selected_loop), 2)
        for layer, dropout in zip(self.concat_layers, self.concat_dropouts):
            x = dropout(self.ELU(layer(x)))
        return x

    def forward(self, tree_tensors):
        tree, comps_first_part, comps_vectors, comps_third_part, loops_tensor, comps_expr_tree = tree_tensors
        device = self.device

        # Expression embedding.
        batch_size, num_comps, len_sequence, len_vector = comps_expr_tree.shape
        x = comps_expr_tree.view(batch_size * num_comps, len_sequence, len_vector)
        _, (expr_embedding, _) = self.exprs_embed(x)
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(batch_size * num_comps, -1)

        # Static computation features.
        batch_size, num_comps, _ = comps_first_part.shape
        first_part = comps_first_part.to(device).view(batch_size * num_comps, -1)
        third_part = comps_third_part.to(device).view(batch_size * num_comps, -1)

        # Affine-transformation embedding.
        vectors = self.encode_vectors(comps_vectors.to(device))
        _, (prog_embedding, _) = self.transformation_vectors_embed(vectors)
        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(batch_size * num_comps, -1)

        # Per-computation embedding.
        x = torch.cat((first_part, prog_embedding, third_part, expr_embedding), dim=1)
        x = x.view(batch_size, num_comps, -1)
        for layer, dropout in zip(self.comp_embedding_layers, self.comp_embedding_dropouts):
            x = dropout(self.ELU(layer(x)))
        comps_embeddings = x

        # Recursive per-root embedding, then aggregate roots.
        roots_list = [
            self.get_hidden_state(root, comps_embeddings, loops_tensor)
            for root in tree["roots"]
        ]
        roots_tensor = torch.cat(roots_list, 1)
        _, (roots_h_n, _) = self.roots_lstm(roots_tensor)
        x = roots_h_n.permute(1, 0, 2)

        for layer, dropout in zip(self.regression_layers, self.regression_dropouts):
            x = dropout(self.ELU(layer(x)))
        out = self.predict(x)
        # Speedups are non-negative; LeakyReLU keeps that without a dead-ReLU region.
        return self.LeakyReLU(out[:, 0, 0])
