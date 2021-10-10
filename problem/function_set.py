import torch
import torch.nn as nn


class NLPFunctionSet:
    @staticmethod
    def return_func_name():
        return [
            {"name": "element_wise_sum", "arity": 2},
            {"name": "element_wise_product", "arity": 2},
            {"name": "concat", "arity": 2},
            {"name": "blending", "arity": 3},
            {"name": "linear", "arity": 1},
            {"name": "sigmoid", "arity": 1},
            {"name": "tanh", "arity": 1},
            {"name": "leaky_relu", "arity": 1},
            {"name": "layer_norm", "arity": 1},
        ]

    @staticmethod
    def return_func_dict():
        return {
            "element_wise_sum": NLPFunctionSet.element_wise_sum,
            "element_wise_product": NLPFunctionSet.element_wise_product,
            "concat": NLPFunctionSet.concat,
            "blending": NLPFunctionSet.blending,
            "linear": NLPFunctionSet.linear,
            "sigmoid": NLPFunctionSet.sigmoid,
            "tanh": NLPFunctionSet.tanh,
            "leaky_relu": NLPFunctionSet.leaky_relu,
            "layer_norm": NLPFunctionSet.layer_norm,
        }

    @staticmethod
    # def element_wise_sum(dim_left, dim_right):
    #    return AddModule(dim_left, dim_right)
    def element_wise_sum(dim):
        return AddModule()

    @staticmethod
    # def element_wise_product(dim_left, dim_right):
    #    return ProductModule(dim_left, dim_right)
    def element_wise_product(dim):
        return ProductModule()

    @staticmethod
    def concat(dim):
        return ConcatModule(dim)

    @staticmethod
    # def blending(dim1, dim2, dim3):
    #    return BlendingModule(dim1, dim2, dim3)
    def blending(dim):
        return BlendingModule()

    @staticmethod
    def linear(dim):
        return nn.Linear(in_features=dim, out_features=dim)

    @staticmethod
    def sigmoid(dim):
        return nn.Sigmoid()

    @staticmethod
    def tanh(dim):
        return nn.Tanh()

    @staticmethod
    def leaky_relu(dim):
        return nn.LeakyReLU()

    @staticmethod
    def layer_norm(dim):
        return nn.LayerNorm(dim)


class ReshapeModule(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        if dim_in != dim_out:
            self.fc = nn.Linear(dim_in, dim_out)
        else:
            self.fc = nn.Identity()

    def forward(self, x):
        x = self.fc(x)
        return x


class AddModule(nn.Module):
    def __init__(self):
        super().__init__()
        # self.reshape = ReshapeModule(dim_left, dim_right)

    def forward(self, a, b):
        # a, b = self.reshape(a, b)
        if a.shape != b.shape:
            print("Shape mismatch")
            print(a.shape)
            print(b.shape)
        x = torch.add(a, b)
        return x


class ProductModule(nn.Module):
    def __init__(self):
        super().__init__()
        # self.reshape = ReshapeModule(dim_left, dim_right)

    def forward(self, a, b):
        # a, b = self.reshape(a, b)
        x = torch.mul(a, b)
        return x


class ConcatModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.reshape = ReshapeModule(dim_in, dim_out)
        self.reshape = ReshapeModule(2 * dim, dim)

    def forward(self, a, b):
        out = torch.cat([a, b], axis=-1)
        out = self.reshape(out)
        return out


# class BlendingModule(nn.Module):
#     def __init__(self, dim_in1, dim_in2, dim_in3):
#         super().__init__()
#         self.product1 = ProductModule(dim_in1, dim_in3)
#         self.product2 = ProductModule(dim_in2, dim_in3)
#         left_dim = max(dim_in1, dim_in3)
#         right_dim = max(dim_in2, dim_in3)
#         self.add = AddModule(left_dim, right_dim)

#     def forward(self, a, b, c):
#         left = self.product1(a, c)
#         right = self.product2(b, 1 - c)
#         return self.add(left, right)


class BlendingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.product1 = ProductModule()
        self.product2 = ProductModule()
        self.add = AddModule()

    def forward(self, a, b, c):
        left = self.product1(a, c)
        neg_c = torch.neg(torch.sub(c, 1))
        right = self.product2(b, neg_c)
        return self.add(left, right)
