---
layout: post
title: "State Monad笔记二"
tags: 
- FP
- haskell
- monad
- "笔记"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

老师：Monad想到什么呢？
同学：毛茸茸的猫猫!

好同学们，请默念“Warm and Fuzzy”100遍~

。。
。。
。。
。
。
。
。
。
。
<a href="http://www.fleurer-lee.com/wp-content/uploads/2010/03/p_large_Pg4a_4600000102612d0c.jpg"><img src="http://www.fleurer-lee.com/wp-content/uploads/2010/03/p_large_Pg4a_4600000102612d0c-300x225.jpg" alt="" title="p_large_Pg4a_4600000102612d0c" width="600" class="alignnone size-medium wp-image-645756" /></a>
。
。
。
。。
。。

100遍了没。。。好，我们先看看State猫猫的类型吧 ^_^

<pre lang="haskell">
newtype State s a = State { runState :: s -> (a, s) }
</pre>

里面有个难看的record syntax，咱们无视之，改成：

<pre lang="haskell">
data State g v = State (g -> (v, g))
</pre>

就是一个lambda，外加一二元组：v表示猫猫正经计算中的值，g就表示猫猫的状态啦(就一个全局变量嘛)~那个lambda的参数g也一样。相应的猫猫Instance也就是如下：

<pre lang="haskell">
instance Monad (State g) where
    return v = 
        State $ \g -> (v, g)

    (>>=) (State m) f = 
        State $ \g -> 
                let (v, g') = m g
                    (State m') = f v
                in  m' g'
</pre>                

还是一环套一环，函数编程没有变量，那就向下传递的时候改变它的值就行了。再加两个小喽骡：

<pre lang="haskell">
get_g = State $ \g -> (g, g)
set_g v = State $ \_ -> (v, v)
</pre>

全局变量的值不也在那个lambda的参数里么，get_g它设成计算中的值，这样在do-notation中就可以g <- get_g了。set_g则是把它的参数设到g里，而原先的g无视扔掉即可。

加个函数试验下：

<pre lang="haskell">
run (State m) i = m i 

add1 :: State Int Int
add1 = do {
    g <- get_g;
    set_g $ g + 1;
}

add3 = do {
    add1;
    add1;
    add1;
}
</pre>

进ghci输入run add3 1可得(4,4)，可见那个“全局变量”就这么变化了。

回头再想想它的类型，一个二元组不就够了嘛，为什么要套那层lambda呢？试下就知道

<pre lang="haskell">
data State g v = State (g, v) 

instance Monad (State g) where
    (>>=) (State (g, v)) f = (g, f v)
</pre>

看着不错，接着来：

<pre lang="haskell">
    return v = ...
</pre>   

发现问题了没，return我们实现不了...搞不到原先g的值，链子就断在了这一环。解决方法就是把它推后，不是没有g么，于是放到参数里等着别人(>>=)给，这算不算惰性呢？ :)

ps:明白了State是怎么回事，附一山寨parsec ^_^ <a href="http://github.com/Fleurer/FParser">http://github.com/Fleurer/FParser</a>
