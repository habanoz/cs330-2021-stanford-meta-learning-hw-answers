"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        self.U = ScaledEmbedding(num_embeddings=num_users, embedding_dim=self.embedding_dim, sparse=sparse)
        self.Q = ScaledEmbedding(num_embeddings=num_items, embedding_dim=self.embedding_dim, sparse=sparse)

        self.A = ZeroEmbedding(num_embeddings=num_users, embedding_dim=1, sparse=sparse)
        self.B = ZeroEmbedding(num_embeddings=num_items, embedding_dim=1, sparse=sparse)

        if embedding_sharing:
            self.U_reg = self.U
            self.Q_reg = self.Q
        else:
            self.U_reg = ScaledEmbedding(num_embeddings=num_users, embedding_dim=self.embedding_dim, sparse=sparse)
            self.Q_reg = ScaledEmbedding(num_embeddings=num_items, embedding_dim=self.embedding_dim, sparse=sparse)

        layer_sizes = list(layer_sizes)
        layers = []
        for i in range(0, len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(layer_sizes[-1], 1))

        self.f = nn.Sequential(*layers)

        # ********************************************************
        # ********************************************************
        # ********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************

        # Matrix Factorization
        u_n = self.U(user_ids)
        q_n = self.Q(item_ids)
        a_n = self.A(user_ids)
        b_n = self.B(item_ids)

        predictions = torch.bmm(u_n.unsqueeze(1), q_n.unsqueeze(1).transpose(1, 2)).squeeze(-1)
        predictions += a_n + b_n
        predictions = predictions.squeeze()

        # Regression
        u_reg_n = self.U_reg(user_ids)
        q_geg_n = self.Q_reg(item_ids)

        f_n = torch.cat((u_reg_n, q_geg_n, u_reg_n * q_geg_n), dim=1)
        score = self.f(f_n)
        score = score.squeeze()

        # ********************************************************
        # ********************************************************
        # ********************************************************
        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")

        assert len(predictions)==len(user_ids)
        assert len(score)==len(user_ids)

        return predictions, score
