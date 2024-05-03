> interacting with a completion LLM is not very natural or useful: you cannot give instructions or ask questions; instead you would always need to formulate your input as a prompt whose natural continuation is your desired output.

IFT: Instruction-Fine Tuning

## Chat Completion Endpoints: under the hood

chat 服务在内部，会将用户的 chat 历史拼接成为一个单独的 string，并拼接一行 `"Assistant: "` 到字符串末尾。

llama 的模型在训练时，用户的输入会包裹在 `[INST]` 和 `[/INST]` 中间。

https://www.reddit.com/r/LocalLLaMA/comments/155po2p/get_llama_2_prompt_format_right/

```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Hi there! 
[/INST] 
Hello! How can I help you today? </s>
<s>[INST] In the text below, find all proper nouns:
    Jack lives in Bosnia, and Jill lives in Belgium.
 [/INST] 
John, Bosnia, Jill, Belgium are all proper nouns. </s><s> 
[INST] Where does Jack live? [/INST] 
Jack lives in Bosnia. </s><s>
[INST] And Jill? [/INST]
Jill lives in Belgium. </s><s>
[INST] Which are its neighboring countries? [/INST]
```


所以 chat ui 需要做的一件事情就是，能够将 chat history 与 prompt 相互转换按照 template 管理好。

比如：

https://github.com/oobabooga/text-generation-webui/tree/main/instruction-templates
