model 的 lifecycle：

1. 打开模型文件，读取 weights 的 shape（暂时未读取 weights 值）
2. 使用读取到的 shape 和元信息，按 tensor 来构造 model struct
3. 编译 model struct 和它的 forward function 到一个 accelerator specific executable；其中 forward 方法中描述了模型计算的数学过程；
4. 从 disk 读取模型 weights，装载到 accelerator memory；
5. 将模型 weights 绑定到 executable；
6. 载入一部分用户输入，拷贝到 accelerator；
7. 根据用户输入执行 executable；
8. 从 accelerator 拷贝输出到宿主机内存；
9. 释放 executable 资源和相关的 weights；

## Tensor Bros.

打交道的不止是 Tensor 一个概念。还有 `Buffer`、`HostBuffer` 和 `Shape`。

- `Shape` 用于描述一个 multi-dimension array；
	- `Shape.init(.{16}, .f32)` 可以创建一个包含 16 个 f32 的 vector
	- `Shape.init(.{512, 1024}, .f16)` 可以创建一个 512 x 1024 的 f16 的矩阵
	- Shape 中只包含 meta data，不关心内存的指向；
	- Shape 也可以表示 scalar，比如 `Shape.init(.{}, .f32)`
- `HostBuffer` ：保存在 CPU 的多维数组；
- `Buffer`：保存在 accelerator 的多维数组，并不能保证这段内存可以为 CPU 访问；可以通过 `zml.aio.loadBuffers` 直接装载到里面；也可以通过 `hostBuffer.toDevice(accelerator)`。
- Tensor：一个数学结构，表示一个计算的中间结果；通常由 `Shape` 和一组表示数学计算过程的 MLIR value 组成；

## The model struct

```
const Model = struct {
    input_layer: zml.Tensor,
    output_layer: zml.Tensor,

    pub fn forward(self: Model, input: zml.Tensor) zml.Tensor {
      const hidden = self.input_layer.matmul(input);
      const output = self.output_layer.matmul(hidden);
      return output;
    }
}
```

Model 中包含 Tensor，只会在编译时有用；

可以通过 [zml.Bufferize(Model)](https://docs.zml.ai/misc/zml_api/#zml.Bufferize(Model)) 将它里面的 Tensor 替换为真实的 buffer；

## Strong type checking

1. Open the model file and read the shapes of the weights -> [zml.HostBuffer](https://docs.zml.ai/misc/zml_api/#zml.HostBuffer) (using memory mapping, no actual copies happen yet)
2. Instantiate a model struct -> `Model` struct (with [zml.Tensor](https://docs.zml.ai/misc/zml_api/#zml.Tensor) inside)
3. Compile the model struct and its `forward` function into an executable. `foward` is a `Tensor -> Tensor` function, executable is a [zml.Exe(Model.forward)](https://docs.zml.ai/misc/zml_api/#zml.Exe(Model.forward))
4. Load the model weights from disk, onto accelerator memory -> [zml.Bufferized(Model)](https://docs.zml.ai/misc/zml_api/#zml.Bufferized(Model)) struct (with [zml.Buffer](https://docs.zml.ai/misc/zml_api/#zml.Buffer) inside
5. Bind the model weights to the executable [zml.ExeWithWeight(Model.forward)](https://docs.zml.ai/misc/zml_api/#zml.ExeWithWeight(Model.forward))
6. Load some user inputs (custom struct), encode them into arrays of numbers ([zml.HostBuffer](https://docs.zml.ai/misc/zml_api/#zml.HostBuffer)), and copy them to the accelerator ([zml.Buffer](https://docs.zml.ai/misc/zml_api/#zml.Buffer)).
7. Call the executable on the user inputs. `module.call` accepts [zml.Buffer](https://docs.zml.ai/misc/zml_api/#zml.Buffer) arguments and returns [zml.Buffer](https://docs.zml.ai/misc/zml_api/#zml.Buffer)
8. Return the model output ([zml.Buffer](https://docs.zml.ai/misc/zml_api/#zml.Buffer)) to the host ([zml.HostBuffer](https://docs.zml.ai/misc/zml_api/#zml.HostBuffer)), decode it (custom struct) and finally return to the user.