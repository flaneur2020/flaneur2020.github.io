
## 3. Naive Kernels

> So, how do we stay sane? We establish a "ground truth." For every GPU kernel we write, we will first write a simple, clear, single-threaded CPU version of the same operation. This CPU version is our oracle. It’s slow, but it’s easy to reason about and trust. After we run our fancy new GPU kernel, our very last step will always be to compare its output, element by element, to the output of our trusted CPU version. If they match, we can proceed with confidence. If not, we know exactly where the problem is.

