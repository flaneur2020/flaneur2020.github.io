

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Gateway   │───▶│ ext-proc     │───▶│ Model Server    │
│  (Envoy)    │    │ Service      │    │ Pods            │
└─────────────┘    └──────────────┘    └─────────────────┘
                          ▲
                          │
                   ┌──────────────┐
                   │ InferencePool│
                   │ Controller   │
                   └──────────────┘
```
## InferencePool

```yaml
apiVersion: inference.x-k8s.io/v1alpha2
kind: InferencePool
metadata:
  name: base-model-pool
spec:
  selector:
    app: llm-server
  targetNumber: 8080
  extensionRef:
    name: infra-backend-v1-app
```

每创建一个 InferencePool，会自动创建一个 gateway extension，它是一个 grpc 服务。可以被 gateway 连接到，去做一些扩展逻辑，比如 Endpoint Picker 这种。

它好像基于 Gateway API 的 extension 机制，做到的扩展能力。内部是基于基于 **Envoy External Processing API** 做的。

这个 extension 机制，可以允许 Gateway 在接到请求时，问一下中间件怎么处理，允许把 header 和 payload 都交给中间件做分流控制。
## InferenceModel

```yaml
apiVersion: inference.x-k8s.io/v1alpha2
kind: InferenceModel
metadata:
  name: npc-bot
spec:
  modelName: npc-bot
  criticality: Critical
  targetModels:
    - name: npc-bot-v1
      weight: 50
    - name: npc-bot-v2
      weight: 50
  poolRef:
    name: base-model-pool
```

每个 Inference Model 有一个 Critical 级别，似乎大约是可不可以被抢占的意思，资源紧张时，可以牺牲 Sheddable 的资源。

每个 Inference Model 的 targetModel 有一个优先级，主要用于 ab test 分流的测试。

同属于一个 Inference Pool 的 Inference Model，最好是同一个模型的不同变种，不要把生图、生文的模型混在一个 Pool 中。

比较建议的用法可以是，同一个 Pool 中放模型的不同 lora：比如 llama-7b-code-lora、llama-7b-translation-lora 等。
## Best Practice

<mark>确保 Inference Pool 内的所有 Inference Model 能够在相同的硬件和软件环境</mark>。


