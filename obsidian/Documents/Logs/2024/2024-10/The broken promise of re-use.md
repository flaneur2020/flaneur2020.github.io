> re-use was accomplished to a lot smaller degree than promised, which meant that the promised business case never got realized

反过来说，搞了 re-use 之后经常出现脆弱、紧耦合的系统。


- First of all: **Never base an architectural decision on re-use**. It will never work and you usually end up at quite the opposite place of where you wanted to go.
- You don’t need any special architectural paradigm for re-use. Everything that is needed for re-use is already in place – batteries included.
- Re-use emerges. **Don’t plan for it**. It takes a lot of work, trial and error and actual experience with real clients to design a good re-usable component which encapsulates the right responsibility and provides it via a good client-driven and easy-to-understand API.
- Re-use is expensive. It takes a lot of effort to make a component re-usable. As a rule of thumb, **it costs about 5 times as much as it costs to make a component usable**. Thus, make sure, it is worth the effort, i.e., the component is used often enough before you make it re-usable.
- And last of all: Don’t go for cheap re-use by building trivial components or layered components. You will end up in a hell where you never, ever wanted to be.

所以该怎么办？

> Strive for replaceability, not re-use. It will lead you the right way.