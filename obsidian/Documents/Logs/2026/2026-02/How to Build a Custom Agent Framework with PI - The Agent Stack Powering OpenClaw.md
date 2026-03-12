
每个 provider 的 streaming 格式不同，pi-ai 的 streamSimple 将它们统一为了：

- start
- text_start
- text_delta
- text_end
- thinking_start/delta/end
- toolcall_start/delta/end
- done
- error

pi-ai 能够支持 switch provider。

## Layer 2: pi-agent-core

Creating an agent:

```
import { Agent } from "@mariozechner/pi-agent-core";
import { getModel, streamSimple } from "@mariozechner/pi-ai";

const model = getModel("anthropic", "claude-opus-4-5");

const agent = new Agent({
  initialState: {
    systemPrompt: "You are a helpful assistant with access to tools.",
    model,
    tools: [weatherTool],
    thinkingLevel: "off",
  },
  streamFn: streamSimple,
});
```