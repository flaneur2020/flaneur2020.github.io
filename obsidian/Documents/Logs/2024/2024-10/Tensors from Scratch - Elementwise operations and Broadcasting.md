
> Expanding and copying the shape and values of a particular tensor BEFORE performing any kind of operation with the modified tensor is called shape broadcasting.

**Algorithm for broadcasting**:

1. Compare the shapes of the input tensors element-wise, starting from the rightmost dimension.
2. If the dimensions are equal, or one of them is 1, move to the next dimension.
3. If the dimensions differ and neither is 1, raise an error.
4. If one tensor has fewer dimensions, prepend 1s to its shape until both tensors have the same number of dimensions.
5. The resulting tensor will have the maximum size along each dimension of the input arrays.
6. For dimensions where one tensor had size 1, that tensor is virtually copied along that dimension to match the other tensor's size.
7. The resultant shape will be the final shape for both the input tensors.