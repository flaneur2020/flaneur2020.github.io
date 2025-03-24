https://simonwillison.net/2025/Feb/21/my-llm-codegen-workflow-atm/#atom-everything

_tl:dr; Brainstorm spec, then plan a plan, then execute using LLM codegen. Discrete loops. Then magic. ✩₊˚.⋆☾⋆⁺₊✧_

## Greenfield
### 1） Idea honing

先按一个头脑风暴来开始一个 greenfield project，目标是生成一个 detailed spec；

```prompt
Ask me one question at a time so we can develop a thorough, step-by-step spec for this idea. Each question should build on my previous answers, and our end goal is to have a detailed specification I can hand off to a developer. Let’s do this iteratively and dig into every relevant detail. Remember, only one question at a time. 
 
Here’s the idea: __IDEA__
```

```prompt
Now that we’ve wrapped up the brainstorming process, can you compile our findings into a comprehensive, developer-ready specification? Include all relevant requirements, architecture choices, data handling details, error handling strategies, and a testing plan so a developer can immediately begin implementation.
```

```prompt
Now that we’ve wrapped up the brainstorming process, can you compile our findings into a comprehensive, developer-ready specification? Include all relevant requirements, architecture choices, data handling details, error handling strategies, and a testing plan so a developer can immediately begin implementation.
```

结果可以保存到项目的 spec.md 中。
### 2) Planning

```prompt
Draft a detailed, step-by-step blueprint for building this project. Then, once you have a solid plan, break it down into small, iterative chunks that build on each other. Look at these chunks and then go another round to break it into small steps. Review the results and make sure that the steps are small enough to be implemented safely with strong testing, but big enough to move the project forward. Iterate until you feel that the steps are right sized for this project.

From here you should have the foundation to provide a series of prompts for a code-generation LLM that will implement each step in a test-driven manner. Prioritize best practices, incremental progress, and early testing, ensuring no big jumps in complexity at any stage. Make sure that each prompt builds on the previous prompts, and ends with wiring things together. There should be no hanging or orphaned code that isn't integrated into a previous step.

Make sure and separate each prompt section. Use markdown. Each prompt should be tagged as text using code tags. The goal is to output prompts, but context, etc is important as well. <SPEC>
```

使用一个 reasoning model 来生成一个 prompt_plan.md。

然后加一个 todo.md 作为更低层的 steps。

```prompt
Can you make a `todo.md` that I can use as a checklist? Be thorough.
```

整个 plan 的过程大约花 15 分钟。

### 3） Execution

I essentially pair program with [claude.ai](https://claude.ai/) and just drop each prompt in iteratively. I find that works pretty well. The back and forth can be annoying, but it largely works.

I am in charge of the initial boilerplate code, and making sure tooling is set up correctly. This allows for some freedom, choice, and guidance in the beginning. Claude has a tendency to just output react code - and having a solid foundation with the language, style, and tooling of your choice will help quite a bit.

I will then use a tool like [repomix](https://github.com/yamadashy/repomix) to iterate when things get stuck (more about that later).

The workflow is like this:

- set up the repo (boilerplate, uv init, cargo init, etc)
- paste in prompt into claude
- copy and paste code from claude.ai into IDE
- run code, run tests, etc
- …
- if it works, move on to next prompt
- if it doesn’t work, use repomix to pass the codebase to claude to debug
- rinse repeat ✩₊˚.⋆☾⋆⁺₊✧

## Non-greenfield: Iteration, incrementally

作者在使用一个叫做 repomix 的工具来将一个仓库打包成对 llm 友好的一个文件。

### Prompt magic

code review：

```prompt
You are a senior developer. Your job is to do a thorough code review of this code. You should write it up and output markdown. Include line numbers, and contextual info. Your code review will be passed to another teammate, so be thorough. Think deeply  before writing the code review. Review every part, and don't hallucinate.
```

GitHub Issue generation

```prompt
You are a senior developer. Your job is to review this code, and write out the top issues that you see with the code. It could be bugs, design choices, or code cleanliness issues. You should be specific, and be very good. Do Not Hallucinate. Think quietly to yourself, then act - write the issues. The issues will be given to a developer to executed on, so they should be in a format that is compatible with github issues
```

missing tests

```prompt
You are a senior developer. Your job is to review this code, and write out a list of missing test cases, and code tests that should exist. You should be specific, and be very good. Do Not Hallucinate. Think quietly to yourself, then act - write the issues. The issues  will be given to a developer to executed on, so they should be in a format that is compatible with github issues
```