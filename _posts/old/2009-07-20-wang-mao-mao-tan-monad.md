---
layout: post
title: "王猫猫谈Monad"
tags: 
- FP
- monad
- "笔记"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

<blockquote><div>
<p class="friend-id">Feather  2009-07-19 21:46:08</p>
<p class="content">世事恰如迷局, 寻人, 失散, 流落, 皈依, 纷乱无绪, 及至谜底解开, 那人却已站立在你面前.</p>

</div></blockquote>

<style type="text/css">
p{
    line-height:18px;
    font-size:14px;
}
p.content, p.system-content{
    padding-left:12px;
    word-wrap:break-word;
    word-break:break-all;
}
p.my-id{
    color:#008040;
}
p.friend-id{
    color:#00f;
}
p.system-id{
    color:#6b6b6b;
}
h3#title {
    font-size:14px;
    font-weight:bold;
}
#chatinfo {
    clear:both;
    color:#808080;
    text-align:left;
    font-size: 80%;
}
#save {
    padding-left:20px;
}
#footer {
    padding-top:30px;
}
#footer img {
    border:0px;
}
</style>


<div>
<p class="friend-id">Feather  2009-07-19 21:46:46</p>
<p class="content">??</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:46:50</p>
<p class="content">继续讲么</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:47:02</p>
<p class="content">没了?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:47:08</p>
<p class="content">别的不知道了...</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:47:15</p>
<p class="content"><img src="/images/Face2/10.gif" alt="" align="absmiddle" />不要啊</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:47:25</p>
<p class="content"><img src="/images/Face2/36.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:47:24</p>
<p class="content">我关于Monad 有一本笔记</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:47:31</p>
<p class="content">扔学校了</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:47:40</p>
<p class="content">拉门</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:47:43</p>
<p class="content">哈哈</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:47:57</p>
<p class="content">讲讲讲讲</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:48:11</p>
<p class="content">从哪开始</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:48:16</p>
<p class="content">你有离散数学基础没?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:48:28</p>
<p class="content">貌似看过一点</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:49:20</p>
<p class="content">哦</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:49:21</p>
<p class="content">那就好</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:49:25</p>
<p class="content">开始讲昂</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:49:34</p>
<p class="content"><img src="/images/Face2/0.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:49:48</p>
<p class="content">first</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:50:00</p>
<p class="content">let talk about some basic concepts</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:50:10</p>
<p class="content">ok</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:50:16</p>
<p class="content">what is type?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:50:21</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:50:35</p>
<p class="content">tell me what you think about type</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:50:38</p>
<p class="content">"type"</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:50:41</p>
<p class="content">what is it</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:50:55</p>
<p class="content">emm...</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:51:02</p>
<p class="content">a system?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:51:59</p>
<p class="content">hold value...operation...and so ... make program easier to be understood by machine?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:52:28</p>
<p class="content">ok~</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:52:35</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:53:13</p>
<p class="content">you just give us the 'Class/Instance" definition in haskell</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:53:41</p>
<p class="content">what's more...<img src="/images/Face2/32.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:53:48</p>
<p class="content">no no no</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:53:56</p>
<p class="content">there's no "what's more'</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:54:02</p>
<p class="content">but what's less.</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:54:10</p>
<p class="content">for short, a type is just a set</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:54:12</p>
<p class="content">set~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:54:14</p>
<p class="content">just set</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:54:16</p>
<p class="content">no more</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:54:19</p>
<p class="content">set~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:54:25</p>
<p class="content">understand?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:54:28</p>
<p class="content">set of integer</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:54:31</p>
<p class="content">set of string</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:54:32</p>
<p class="content">ya~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:54:35</p>
<p class="content">set of Monad</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:54:38</p>
<p class="content">haha ~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:55:01</p>
<p class="content">understand?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:55:35</p>
<p class="content">maybe mis the point , but Monad is a set of types?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:56:01</p>
<p class="content">forget my saying "set of Monad" first</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:56:16</p>
<p class="content">ramen</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:56:33</p>
<p class="content">ya</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:56:39</p>
<p class="content">plz go on :&gt;</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:56:46</p>
<p class="content">what is function?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:56:58</p>
<p class="content">emmm</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:57:08</p>
<p class="content">a map between sets?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:57:13</p>
<p class="content">D -&gt; R</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:57:25</p>
<p class="content">Int -&gt; String</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:57:42</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:57:43</p>
<p class="content">a set of combination (a,b)~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:57:45</p>
<p class="content">yes?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:58:02</p>
<p class="content">let's take a simple e.g.</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:58:07</p>
<p class="content">Relation?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:58:08</p>
<p class="content">the function `inc`</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:58:45</p>
<p class="content">for clear definithion of `function`, refer to some book</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:58:46</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:59:06</p>
<p class="content">inc = { ..., (0,1), (1,2), (2,3).....}</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:59:09</p>
<p class="content">yes?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:59:16</p>
<p class="content">so we give `inc 2`</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:59:20</p>
<p class="content">we got 3</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 21:59:33</p>
<p class="content">ya</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 21:59:55</p>
<p class="content">lalala ~~~~</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:00:12</p>
<p class="content">so?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:00:29</p>
<p class="content">so~ a monoid~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:00:35</p>
<p class="content">what is monoid</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:00:45</p>
<p class="content">monoid<img src="/images/Face2/32.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:00:45</p>
<p class="content">a set, whith some operation</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:01:18</p>
<p class="content">then what's the difference between a monoid &amp; a function?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:01:37</p>
<p class="content">and difference between a monoid &amp; dict?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:01:43</p>
<p class="content">~~~~~~~~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:01:45</p>
<p class="content">wrong</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:01:54</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:01:55</p>
<p class="content">consider this</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:02:08</p>
<p class="content">ya</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:02:48</p>
<p class="content">a monoid is a set M, together with an operation "●" that combines any two elements a and b to form another element denoted a ● b.</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:02:57</p>
<p class="content">formal definition here</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:03:14</p>
<p class="content">● is called product</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:03:28</p>
<p class="content">multiply?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:05:09</p>
<p class="content">orz ,forget my last word</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:05:17</p>
<p class="content">sorry</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:05:18</p>
<p class="content">plz go on :&gt;</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:05:47</p>
<p class="content">the pc got slow...</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:05:53</p>
<p class="content">ya?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:05:56</p>
<p class="content">because i am compiling...</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:06:21</p>
<p class="content">compiling is ram hungry...</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:06:28</p>
<p class="content">for e.g</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:06:34</p>
<p class="content">int, *</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:06:41</p>
<p class="content">in the set int</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:06:48</p>
<p class="content">we define *</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:07:02</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:07:24</p>
<p class="content">here we can see what the monoid is</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:07:47</p>
<p class="content">3 important axioms</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:07:49</p>
<p class="content">oui?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:07:51</p>
<p class="content">1&gt;</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:07:54</p>
<p class="content">closure</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:08:05</p>
<p class="content">int * int is a int , not other thing</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:08:13</p>
<p class="content">2&gt; Associatiity</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:09:09</p>
<p class="content">for int a,b,c</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:09:23</p>
<p class="content">(a*b)*c equals to a*(b*c)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:09:26</p>
<p class="content">see?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:09:33</p>
<p class="content">got it</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:10:21</p>
<p class="content">3rd?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:10:21</p>
<p class="content">3&gt;</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:11:05</p>
<p class="content">the e, so called `one`</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:11:21</p>

<p class="content">for any int a
e * a = a * e = a

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:11:34</p>
<p class="content">note, my e.g. is not very good</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:12:14</p>
<p class="content">so e here is the identity element</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:12:35</p>
<p class="content">like 1 in Int?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:13:09</p>
<p class="content">yes</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:13:33</p>
<p class="content">linear?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:14:22</p>
<p class="content">[Int] is a Monoid</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:14:44</p>
<p class="content">the product is  ++</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:14:51</p>
<p class="content">the id is?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:14:54</p>
<p class="content">tell me</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:15:05</p>
<p class="content">[]?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:15:18</p>
<p class="content">yes~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:15:33</p>
<p class="content">so what is Functor</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:15:41</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:16:18</p>
<p class="content">...er...</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:16:26</p>
<p class="content">not this fast</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:16:42</p>
<p class="content">oui</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:17:00</p>
<p class="content">first let's talk about category</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:17:52</p>
<p class="content">ya</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:18:17</p>
<p class="content">what is category</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:18:23</p>
<p class="content">？</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:18:28</p>
<p class="content">so many whats... :)</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:19:01</p>
<p class="content">heihiehie</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:19:42</p>
<p class="content">now you know what monoid is</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:19:57</p>
<p class="content">let's consider  category</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:21:16</p>
<p class="content">yeah</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:21:27</p>
<p class="content">so slow a computer is ....</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:21:43</p>
<p class="content">compiling eats computer...</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:22:36</p>
<p class="content">a category...</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:22:57</p>
<p class="content">=</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:23:22</p>
<p class="content">objects + arrows</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:23:25</p>
<p class="content">？</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:23:53</p>
<p class="content">what is arrow here?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:24:10</p>
<p class="content">-&gt; ?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:24:17</p>
<p class="content">fuck</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:24:22</p>
<p class="content">no haskell code here</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:24:42</p>
<p class="content">so ... just an arrow?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:24:59</p>
<p class="content">haha ~ good</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:25:06</p>
<p class="content">it is ----------------------&gt;</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:25:17</p>
<p class="content">-------------------------------------------------------------------&gt;</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:25:30</p>
<p class="content">arrow is just an arrow from an object to an object</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:25:40</p>
<p class="content">map?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:25:43</p>
<p class="content">consider this,</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:25:55</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:26:32</p>
<p class="content">objects = 1,2,3,4,5,6....</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:26:50</p>
<p class="content">an infinite set?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:27:10</p>
<p class="content">arrow = 1-&gt;2, 3-&gt;3, 4-&gt;5, 5-&gt;6</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:27:20</p>
<p class="content">call it whatever you like ~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:27:31</p>
<p class="content">they are objects</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:27:43</p>
<p class="content">and wo got arrows 1-&gt;2, 3-&gt;3, 4-&gt;5, 5-&gt;6 ...</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:27:50</p>
<p class="content">er.... so .... map among themselves?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:28:02</p>
<p class="content">understand what object and arrow is?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:28:25</p>
<p class="content">here we define an action~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:28:28</p>
<p class="content">coposite</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:28:30</p>
<p class="content">yea.... maybe</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:28:43</p>
<p class="content">composite, fuck , spell ....</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:29:23</p>

<p class="content">objects = 1,2,3,4,5,6....
arrows 1-&gt;2, 3-&gt;3, 4-&gt;5, 5-&gt;6 ...

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:29:37</p>
<p class="content">so ?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:30:24</p>
<p class="content">(1-2)◎(2-&gt;3)    we got   1-&gt;3</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:30:28</p>
<p class="content">haha ~ see?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:30:34</p>
<p class="content">this is composite</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:31:28</p>
<p class="content">like . in haskell?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:31:43</p>
<p class="content">fuck, i said no more haskell code</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:31:52</p>
<p class="content">ramen</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:31:57</p>
<p class="content">it is composite in category</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:31:58</p>
<p class="content">here</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:32:09</p>
<p class="content">so ... composite makes a link?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:32:13</p>
<p class="content">(1-2)◎(2-&gt;3) ~ this is composite</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:33:17</p>
<p class="content">ya</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:34:28</p>
<p class="content">((1-2)◎(2-&gt;3))◎(3-&gt;4)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:34:35</p>
<p class="content">what is it?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:34:47</p>
<p class="content">two composites?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:35:03</p>
<p class="content">and compare with</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:35:04</p>
<p class="content">(1-2)◎( (2-&gt;3)◎(3-&gt;4) )</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:35:27</p>
<p class="content">they equal?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:35:31</p>
<p class="content">yes</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:35:39</p>
<p class="content">so recall what monoid is</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:35:43</p>
<p class="content">assoc</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:35:49</p>
<p class="content">this is called~ associativity</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:35:50</p>
<p class="content">yes</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:35:53</p>
<p class="content">clever</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:35:55</p>
<p class="content">~!</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:35:59</p>
<p class="content">haha</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:36:01</p>
<p class="content">orz, i can't recall the word..窘</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:36:26</p>
<p class="content">we have assoc... here</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:37:22</p>
<p class="content">and for every object</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:37:27</p>
<p class="content">we define</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:37:34</p>
<p class="content">ye？</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:37:36</p>
<p class="content">for every object  a</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:37:44</p>
<p class="content">we define Ia</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:38:06</p>
<p class="content">to identify itself?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:38:37</p>
<p class="content">I3 ◎ (3-&gt;2) ====== (3-&gt;2)◎I3 ======= (3-&gt;2)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:39:06</p>
<p class="content">we also have I1, I2,  I3, Ix.....</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:39:11</p>
<p class="content">a start point ?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:39:23</p>
<p class="content">this is called identity</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:39:30</p>
<p class="content">identity in ◎</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:40:22</p>
<p class="content">I3 ◎ (3-&gt;2) ====== (3-&gt;2)◎I2 ======= (3-&gt;2)</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:40:28</p>
<p class="content">oh ,got it . like I3, identify this object is 3?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:40:50</p>
<p class="content">I3 * (3 -&gt; x) ====== =3-&gt;x</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:41:01</p>
<p class="content">here we use ints as objects</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:41:23</p>
<p class="content">yeah?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:41:41</p>
<p class="content">use `identity arrow`</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:41:54</p>
<p class="content">composite another arrow</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:41:59</p>
<p class="content">what we got?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:42:08</p>
<p class="content">a link?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:42:09</p>
<p class="content">the arrow unchanged</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:42:11</p>
<p class="content">fuck</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:42:35</p>
<p class="content">we talk about arrow</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:42:42</p>
<p class="content">窘</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:42:43</p>
<p class="content">why you always link, link, link</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:42:50</p>
<p class="content">what do you want to link?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:42:54</p>
<p class="content">spring brother?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:43:10</p>
<p class="content">BS</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:43:13</p>
<p class="content">ramen, i've made a mistake</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:43:20</p>
<p class="content">let's go on</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:43:23</p>
<p class="content">category</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:43:28</p>
<p class="content">never link anymore</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:43:28</p>
<p class="content">understand it?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:43:35</p>
<p class="content">ya</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:43:51</p>
<p class="content">another e.g.</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:44:14</p>

<p class="content">objects :       a, b, c
arrows:     a-&gt;b,  b-&gt;c, a-&gt;c

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:44:35</p>
<p class="content">(a-&gt;b) ◎ (b-&gt;c)  ==========(a-&gt;c)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:44:53</p>
<p class="content">note here, in math ◎ is not the  . in haskell</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:45:02</p>
<p class="content">oui</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:45:03</p>
<p class="content">the operator order is different</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:45:18</p>
<p class="content">go on</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:45:25</p>

<p class="content">objects :       a, b, c
arrows:     a-&gt;b,  b-&gt;c, a-&gt;c

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:45:41</p>
<p class="content">what else we need to make it a category?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:45:55</p>
<p class="content">answer my question</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:46:15</p>
<p class="content">er...morsphy or what's the word called?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:47:05</p>
<p class="content">morphism</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:47:06</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:47:09</p>
<p class="content">is this word?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:47:14</p>
<p class="content">oui?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:47:18</p>
<p class="content">answer my question</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:47:32</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:47:49</p>

<p class="content">objects :       a, b, c
arrows:     a-&gt;b,  b-&gt;c, a-&gt;c
what else we need to make it a category?

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:48:28</p>
<p class="content">could not tell that...orz</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:48:51</p>
<p class="content">recall what i told you</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:49:04</p>
<p class="content">objects, arrows, associativity, identity</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:49:09</p>
<p class="content">er....</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:49:09</p>
<p class="content">which?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:49:21</p>
<p class="content">associativity?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:49:38</p>
<p class="content">since we have a-&gt;b,  b-&gt;c, a-&gt;c</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:49:55</p>
<p class="content">we alreay got associativity</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:50:08</p>
<p class="content">what we really need here is identity</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:50:22</p>
<p class="content">we have no identity arrows for  a,b,c</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:50:23</p>
<p class="content">oui?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:50:24</p>
<p class="content">right?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:50:25</p>
<p class="content">see?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:50:28</p>
<p class="content">understand?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:50:34</p>
<p class="content">oui</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:51:12</p>
<p class="content">understand?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:51:57</p>
<p class="content">ya</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:52:09</p>
<p class="content">hehe</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:52:11</p>
<p class="content">next</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:52:20</p>
<p class="content">we have category here</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:52:21</p>
<p class="content">haha</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:52:35</p>
<p class="content"><img src="/images/Face2/30.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:53:36</p>
<p class="content">so how?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:53:36</p>

<p class="content">in haskell ,
{id, read, show, Int, Char }

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:53:43</p>
<p class="content">is a category</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:53:55</p>
<p class="content">objects : Int, Char</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:53:59</p>
<p class="content">right?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:54:08</p>
<p class="content">ya</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:54:08</p>
<p class="content">Char -&gt; String</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:54:13</p>
<p class="content">not char</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:54:17</p>
<p class="content">it's string</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:54:18</p>
<p class="content">sorry</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:54:22</p>
<p class="content">mistake here</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:54:24</p>
<p class="content">haha</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:54:25</p>
<p class="content">got it</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:54:26</p>
<p class="content">forgive me</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:54:42</p>
<p class="content"><img src="/images/Face2/0.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:54:55</p>
<p class="content">why it is a category</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:55:04</p>
<p class="content">{id, read, show, Int, String }</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:55:10</p>
<p class="content">yeah</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:55:21</p>
<p class="content">object : Int , String</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:55:27</p>
<p class="content">yes yes</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:55:30</p>
<p class="content">arrows: read ,show</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:55:36</p>
<p class="content">no no</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:55:38</p>
<p class="content">id: identifier</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:55:44</p>
<p class="content"><img src="/images/Face2/1.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:55:47</p>
<p class="content">arrows contains identity</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:55:55</p>
<p class="content">so just put id in arrows</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:56:12</p>
<p class="content">what you said is right too</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:56:16</p>
<p class="content">hehe</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:56:21</p>
<p class="content">y<img src="/images/Face2/13.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:56:47</p>
<p class="content">so?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:57:14</p>
<p class="content">so you've mastered category more or less</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:57:30</p>
<p class="content">oui?<img src="/images/Face2/32.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:57:33</p>
<p class="content">arrows: read ,show, id ~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:57:49</p>
<p class="content">whats the type of them</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:57:59</p>
<p class="content">read :: String -&gt; Int</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:58:05</p>
<p class="content">show :: Int -&gt; String</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:58:13</p>
<p class="content">id :: a-&gt;a?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:58:14</p>
<p class="content">what is `id`?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:58:16</p>
<p class="content">fuck</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:58:28</p>
<p class="content">orz</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:58:34</p>
<p class="content">I said what? for every object, there is ...</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:58:37</p>
<p class="content">so</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:58:45</p>
<p class="content">id(int) :: Int -&gt; Int</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:58:51</p>
<p class="content">T_T</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:58:53</p>
<p class="content">id(string) :: String-&gt; String</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:58:55</p>
<p class="content">哈哈</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:59:04</p>
<p class="content">GO ON bs</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 22:59:10</p>
<p class="content">jiong</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:59:18</p>
<p class="content">did you  get it?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:59:21</p>
<p class="content">I mean</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:59:34</p>
<p class="content">identity is many arrows, not a single arrow</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:59:52</p>
<p class="content">in haskell, we use id::a-&gt;a to name every identity arrows</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 22:59:56</p>
<p class="content">but in math</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:00:01</p>
<p class="content">emm, id is closed in this category?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:00:00</p>
<p class="content">this is different</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:00:17</p>
<p class="content">all arrows must closed in category</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:00:44</p>
<p class="content">got it <img src="/images/Face2/13.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:00:54</p>
<p class="content">you can't use arrow to form another object which is not in the objects</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:01:04</p>
<p class="content">clever</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:01:06</p>
<p class="content">next</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:01:12</p>
<p class="content">a great thing</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:01:18</p>
<p class="content">oui!</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:01:22</p>
<p class="content">Functor</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:01:44</p>
<p class="content">Functor is what?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:01:47</p>
<p class="content">another what~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:01:48</p>
<p class="content">haha</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:02:02</p>
<p class="content">ramen</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:02:59</p>
<p class="content">haha</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:03:14</p>
<p class="content">functor is a morphism from category to category</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:03:24</p>
<p class="content"><img src="/images/Face2/32.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:03:42</p>
<p class="content">how?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:03:46</p>
<p class="content">easy</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:03:53</p>
<p class="content">:D</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:04:17</p>

<p class="content">(2009-07-19 22:47:56)   Feather(85660100)
objects :       a, b, c
arrows:     a-&gt;b,  b-&gt;c, a-&gt;c , and its ids~

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:04:49</p>
<p class="content">we invent a functor F</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:04:58</p>
<p class="content">F must do 2 things</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:05:19</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:05:21</p>
<p class="content">for e.g. F :: Category A -&gt; Category B</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:05:38</p>
<p class="content">F must transform objects in A to objects in B</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:05:47</p>
<p class="content">and arrows in A to arrows in B</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:05:50</p>
<p class="content">right?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:06:00</p>
<p class="content">because functor is a morphism from category to category</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:06:17</p>
<p class="content">yeah</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:06:23</p>
<p class="content">but how?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:06:31</p>
<p class="content">how to transform both?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:06:36</p>

<p class="content">objects :       a, b, c
arrows:     a-&gt;b,  b-&gt;c, a-&gt;c , and its ids~
we name it to `A`

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:06:55</p>
<p class="content">then we define category B</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:07:02</p>
<p class="content">oui</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:07:08</p>

<p class="content">object : d
arrow : d-&gt;d

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:07:38</p>
<p class="content">a super monad isn't it?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:07:47</p>
<p class="content">oui?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:07:52</p>
<p class="content">so F :: Cat A -&gt; Cat B</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:08:20</p>

<p class="content">just transform all objects, a,b,c to d
all arrows x-&gt;x to a single d-&gt;d

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:08:34</p>
<p class="content">wait ....</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:08:36</p>
<p class="content">then we got F, and implement it</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:08:57</p>
<p class="content">bind :: M a -&gt; M b</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:09:12</p>
<p class="content">fuck, no more haskell code here</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:09:15</p>
<p class="content">orz, forget it</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:09:22</p>
<p class="content">fuck to to death</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:09:26</p>
<p class="content">plz go &amp; on</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:09:39</p>
<p class="content">orz orz . i 've made a silly mistake</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:09:51</p>
<p class="content">no more no more any</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:09:55</p>
<p class="content">bind is not Functor</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:10:09</p>
<p class="content">yeah</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:10:15</p>
<p class="content">now, have you mastered F :: Category A -&gt; Category B</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:10:22</p>
<p class="content">it is a Functor</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:10:23</p>
<p class="content">^_^</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:10:40</p>
<p class="content">you can easily prove it ,</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:10:51</p>
<p class="content">noice that</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:11:14</p>
<p class="content">here. a functor preseve identity</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:11:30</p>
<p class="content"><img src="/images/Face2/32.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:11:45</p>
<p class="content">just F :: Category A -&gt; Category B , we transform all arrows to the id(d)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:11:52</p>
<p class="content">so identity preseves</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:12:13</p>
<p class="content">if more arrows and more objects here</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:12:39</p>
<p class="content">F will transform any id in A into id in B</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:12:50</p>
<p class="content">understand?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:13:18</p>
<p class="content">emm....</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:14:09</p>
<p class="content">'ve got what it is. but still couldn't get how...</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:14:45</p>
<p class="content">F(a-&gt;b) ◎ F(b-&gt;c)    =====   F ( (a-&gt;b)◎(b-&gt;c) )</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:14:46</p>
<p class="content">first</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:15:08</p>
<p class="content">it preserves composition</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:15:11</p>
<p class="content">and identity</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:16:01</p>
<p class="content">preserves...</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:16:19</p>
<p class="content">F( id(a) ◎ F(a-&gt;c) )  =====   id(d)  ◎  F( (a-&gt;c) )</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:16:48</p>
<p class="content">F( id(a) ◎ F(a-&gt;c) )  =====   id(d)  ◎  F( (a-&gt;c) )    ======= F( a-&gt; c ) ==== d-&gt;d</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:17:26</p>
<p class="content">oui</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:17:34</p>
<p class="content">understand?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:17:50</p>
<p class="content">not very</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:18:04</p>
<p class="content">oh silly am I</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:18:09</p>
<p class="content">Functor is a transformation</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:18:18</p>
<p class="content">from a category to another category</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:18:22</p>
<p class="content">right?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:18:29</p>
<p class="content">yeah</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:18:40</p>
<p class="content">and we know that category have objects, arrows, ids, compostions</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:18:51</p>
<p class="content">so when transformming</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:19:13</p>
<p class="content">the result must satisfy with these</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:19:25</p>
<p class="content">also have objects, arrows, ids, compostions</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:19:35</p>
<p class="content">oui</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:20:14</p>

<p class="content">F(a-&gt;b) ◎ F(b-&gt;c)    =====   F ( (a-&gt;b)◎(b-&gt;c) )
when we have this, we can prove all arrow transform preserve compostion law

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:20:15</p>
<p class="content">right?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:20:33</p>
<p class="content">yeah</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:20:45</p>

<p class="content">F( id(a) ◎ F(a-&gt;c) )  =====   id(d)  ◎  F( (a-&gt;c) )    ======= F( a-&gt; c ) ==== d-&gt;d
and with this low

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:21:01</p>
<p class="content">all identity preserves</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:21:10</p>
<p class="content">because my e.g. is too simple</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:21:24</p>
<p class="content">F( id(a) ◎ F(a-&gt;c) )  =====   id(d)  ◎  F( (a-&gt;c) )</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:21:26</p>
<p class="content">got it partly</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:21:27</p>
<p class="content">just this</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:21:33</p>
<p class="content">F( id(a) ◎ F(a-&gt;c) )  =====   id(d)  ◎  F( (a-&gt;c) )</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:22:04</p>
<p class="content">oui!</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:22:05</p>
<p class="content">you can see that F will got change identity</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:22:21</p>
<p class="content">transform an id we got id in another category</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:22:23</p>
<p class="content">right?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:22:26</p>
<p class="content">got it?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:22:35</p>
<p class="content">yeah</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:22:58</p>
<p class="content">no deep understanding~ just make sence of it</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:23:12</p>
<p class="content">haha</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:23:29</p>
<p class="content">ok~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:23:35</p>
<p class="content">we got to funcor now</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:23:44</p>
<p class="content">oui</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:23:49</p>
<p class="content">so many ouis</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:23:49</p>
<p class="content">and the greatest and most powerful</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:23:55</p>
<p class="content">mo....</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:23:56</p>
<p class="content">the Monad~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:23:58</p>
<p class="content">yes</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:24:02</p>
<p class="content">yei!</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:24:02</p>
<p class="content">what is Monad~</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:24:15</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:24:15</p>
<p class="content">it's the most fucking question</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:24:18</p>
<p class="content">right?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:24:19</p>
<p class="content">hehe</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:24:32</p>
<p class="content"><img src="/images/Face2/36.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:24:51</p>
<p class="content">so, remember~ I said to you that~ Monad is just Functor</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:25:23</p>
<p class="content">here no need to think it in haskell method~ just remember</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:25:42</p>
<p class="content">F :: Cat A -&gt; Cat B</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:25:44</p>
<p class="content">Monad is also a transform from category to another category...</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:25:46</p>
<p class="content">yes yes</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:25:49</p>
<p class="content">recall it</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:26:00</p>
<p class="content">and then I will tell you more</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:26:09</p>
<p class="content">A has objects &amp; arrows</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:26:10</p>
<p class="content">F :: Cat A -&gt; Cat B is a Functor</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:26:18</p>
<p class="content">transfrom it into B</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:26:25</p>
<p class="content">OK,~ let's begin</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:26:36</p>
<p class="content">ya</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:26:38</p>
<p class="content">F :: Cat A -&gt; Cat B is a Functor , but we cam't treat it as a monad</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:26:43</p>
<p class="content">what is Monad~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:26:49</p>
<p class="content">A endofunctor</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:26:56</p>
<p class="content">endo?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:26:56</p>
<p class="content">what is endofunctor~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:27:06</p>
<p class="content">F :: Cat A -&gt; Cat A is endofunctor</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:27:11</p>
<p class="content">got it?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:27:18</p>
<p class="content">transform it to itself</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:27:19</p>
<p class="content">haha</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:27:44</p>
<p class="content">oui, i've got the difference...maybe</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:27:51</p>
<p class="content">what's more</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:28:09</p>
<p class="content">don't worry, later i'll show you how</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:28:11</p>
<p class="content">and why</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:28:21</p>
<p class="content">:)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:28:28</p>
<p class="content">endofunctor is a functor from cat A to cat A</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:28:32</p>
<p class="content">the same Category</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:28:45</p>
<p class="content">A monad is an endofunctor</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:28:50</p>
<p class="content">but not so simple</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:28:54</p>
<p class="content">so ?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:29:19</p>
<p class="content">with 2 function</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:29:29</p>
<p class="content">bind return?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:29:36</p>
<p class="content">or you can call the 2 function `natural transformaton`</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:29:37</p>
<p class="content">fuck</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:29:43</p>
<p class="content">orz</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:29:44</p>
<p class="content">no any haskell code</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:29:55</p>
<p class="content">no any more</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:30:09</p>
<p class="content">μ η</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:30:16</p>
<p class="content">orz</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:30:23</p>
<p class="content">holy GREEK</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:30:32</p>
<p class="content">haha~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:30:43</p>
<p class="content">I'll not use μ η here</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:30:58</p>
<p class="content">thanks Spring brother</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:31:42</p>
<p class="content">we define these</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:32:39</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:34:38</p>
<p class="content">~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:34:45</p>
<p class="content">here</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:34:48</p>
<p class="content">oui</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:34:49</p>
<p class="content">let's begin</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:34:57</p>
<p class="content">two function~</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:34:58</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:35:14</p>
<p class="content">unit,  and  join</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:35:28</p>
<p class="content">much better names</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:35:55</p>
<p class="content">I think we can go back to Haskell</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:36:05</p>
<p class="content">oui</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:36:16</p>
<p class="content">what is unit and join? leave it unsolved</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:36:30</p>
<p class="content">ok</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:36:32</p>
<p class="content">answer my question, what is Functor in Haskell</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:36:42</p>
<p class="content">we now talk Haskell code</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:36:43</p>
<p class="content">monad?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:36:52</p>
<p class="content">orz</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:36:57</p>
<p class="content">a class</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:36:57</p>
<p class="content">what is Functor in Haskell</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:37:00</p>
<p class="content">yes</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:37:04</p>
<p class="content">what class</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:37:06</p>

<p class="content">class Functor where
fmap

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:37:12</p>
<p class="content">yes ~</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:37:14</p>
<p class="content">just a fmap?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:37:19</p>
<p class="content">haha</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:37:25</p>
<p class="content">you feel strange</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:37:36</p>
<p class="content">yeah</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:37:40</p>
<p class="content">why it is so different from our math definition~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:37:44</p>
<p class="content">is it?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:37:57</p>
<p class="content">yes!</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:38:08</p>
<p class="content">recall~ Functor here</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:38:11</p>
<p class="content">a Functor...</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:38:18</p>
<p class="content">transfomr arrows, objects</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:38:24</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:38:32</p>
<p class="content">so what is the type of fmap</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:39:01</p>
<p class="content">just give me haskell code</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:39:06</p>
<p class="content">fmap :: (a-&gt;b) -&gt;M a -&gt; M b</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:39:09</p>
<p class="content">yes~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:39:20</p>
<p class="content">you can see</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:39:31</p>
<p class="content">is just transform arrows</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:39:34</p>
<p class="content">it is?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:40:01</p>
<p class="content">arrows?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:40:04</p>
<p class="content">By the way ~ do you know what is the category in haskell ?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:40:19</p>
<p class="content">oh no</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:40:25</p>
<p class="content">fmap just transform arrows</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:40:32</p>
<p class="content">itn't it?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:40:51</p>
<p class="content">arrows....</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:41:01</p>
<p class="content">that is functions in haskell</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:41:06</p>
<p class="content">just functions</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:41:07</p>
<p class="content">haha</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:41:23</p>
<p class="content">what is the category in haskell ?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:41:25</p>
<p class="content">but ...</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:41:33</p>
<p class="content">we have Functor</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:41:41</p>
<p class="content">but what is the category?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:41:50</p>
<p class="content">haha ~ do you want to ask this?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:41:51</p>
<p class="content">orz, monad?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:42:40</p>
<p class="content">when we talk Functor, we must have to categories. yes?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:43:17</p>
<p class="content">yes, but this Functor is NOT that functor?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:43:24</p>
<p class="content">no no</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:43:33</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:43:34</p>
<p class="content">haskell is very math-strict</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:43:53</p>
<p class="content">Functor in haskell is Functor</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:44:02</p>
<p class="content">remeber this:</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:44:07</p>
<p class="content">?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:44:20</p>
<p class="content">in haskell what we working on is a big category</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:44:26</p>
<p class="content">we call it H</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:44:29</p>
<p class="content">`H`</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:44:37</p>
<p class="content">wow</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:44:42</p>
<p class="content">it objects is all types in Haskell</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:44:53</p>
<p class="content">it arrows is all functions in Haskell</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:45:06</p>
<p class="content">its identiry is the super function id</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:45:17</p>
<p class="content">yeah</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:45:21</p>
<p class="content">id: a-&gt;a, so we have all identiry form it</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:45:26</p>
<p class="content">understand?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:45:32</p>
<p class="content">got it</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:45:33</p>
<p class="content">so ~ so~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:45:38</p>
<p class="content">fmap</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:45:47</p>
<p class="content">fmap :: (a-&gt;b) -&gt;M a -&gt; M b</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:46:11</p>
<p class="content">tansform a function a-&gt; b to a function in (M H)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:46:16</p>
<p class="content">isn't it?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:46:22</p>
<p class="content">we just call it M H</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:46:50</p>

<p class="content">a -&gt; b is in the category H
M a -&gt; M b in (M H)

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:46:56</p>
<p class="content">am I right?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:47:34</p>
<p class="content">the category changed into H?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:47:55</p>
<p class="content">H is the global haskell category</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:48:05</p>
<p class="content">so a-&gt;a function(arrow) in in it</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:48:20</p>
<p class="content">and we wrap it in (M H) category</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:48:37</p>
<p class="content">oui, catch it</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:48:42</p>
<p class="content">notice that type is object</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:49:03</p>
<p class="content">arrow from object to object, ~ just from type to type, haha</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:49:15</p>
<p class="content">:D</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:49:24</p>
<p class="content">so by using fmap, we transform all arrows(functions)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:49:28</p>
<p class="content">right?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:49:43</p>
<p class="content">my mind was brighted :D</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:49:50</p>
<p class="content">but</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:49:59</p>
<p class="content">the object transformation?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:50:05</p>
<p class="content">what is it?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:50:11</p>
<p class="content">oui , that's it</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:50:27</p>
<p class="content">so it comes bind and return ?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:50:35</p>
<p class="content">in haskell's Functor definition.. there's no object transformation</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:50:37</p>
<p class="content">fuck</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:50:42</p>
<p class="content">orz</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:50:46</p>
<p class="content">forget it</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:50:48</p>
<p class="content">what did I say</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:50:54</p>
<p class="content">arrow, from type to type</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:51:01</p>
<p class="content">object , is type</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:51:13</p>
<p class="content">so transform object is transform type</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:51:20</p>
<p class="content">right?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:51:24</p>
<p class="content">got it</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:51:48</p>

<p class="content">fmap :: (a-&gt;b) -&gt;M a -&gt; M b
fuck this type definition carefully

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:52:01</p>
<p class="content">the big `M` is what?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:52:05</p>
<p class="content">orz</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:52:13</p>
<p class="content">isn't it what we want</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:52:21</p>
<p class="content">Int -&gt; M Int</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:52:25</p>
<p class="content">MMMonad</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:52:38</p>
<p class="content">M just transform the type to another type</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:52:42</p>
<p class="content">right?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:52:49</p>
<p class="content">yeah</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:53:03</p>
<p class="content">so I said, M is type constructor in haskell, it transforms the object</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:53:14</p>
<p class="content">that is to say, transforms the type</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:53:15</p>
<p class="content">haha</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:53:20</p>
<p class="content">understand?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:53:22</p>
<p class="content">orz, return comes into my mind again</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:53:43</p>
<p class="content">return is just a single function, from a value to another value</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:53:52</p>
<p class="content">not Functor transformation</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:53:57</p>
<p class="content">oui, rememberd</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:54:05</p>
<p class="content">haha, so return is not</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:54:12</p>
<p class="content">what we need is a type constructor</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:54:13</p>
<p class="content">it is the big M</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:54:15</p>
<p class="content">yes</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:54:17</p>
<p class="content">not a function</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:54:35</p>
<p class="content">function can only transform a value</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:54:39</p>
<p class="content">not a whole type</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:54:44</p>
<p class="content">got it?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:55:03</p>
<p class="content">so for begginers, the can't tell from return and the big `M`</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:55:12</p>
<p class="content">but type constructor can do the transformation between types -- so as objects</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:55:27</p>
<p class="content">so for e.g</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:55:33</p>
<p class="content">:D</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:55:39</p>
<p class="content">from Int to Maybe Int</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:55:51</p>
<p class="content">how can we translate Int to Maybe Int</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:55:59</p>
<p class="content">it is the Type Constructor</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:56:03</p>
<p class="content">`Maybe`</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:56:06</p>
<p class="content">not return</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:56:13</p>
<p class="content">yay</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:56:16</p>
<p class="content">return Int ??? fuck this~</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:56:22</p>
<p class="content">haha</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:56:25</p>
<p class="content">silly</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:56:34</p>
<p class="content">object ~ type~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:56:45</p>
<p class="content">remeber, type is just sets</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:57:02</p>
<p class="content">oui</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:57:06</p>
<p class="content">type Int is just {....-1, 0, 1,2, 3...}</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:57:21</p>
<p class="content">understand the whole fucking thing?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:57:23</p>
<p class="content">haha, just the same as you said</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:57:53</p>
<p class="content">so Functor in Haskell is just simple Functor~ isn't~?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:58:14</p>
<p class="content">ya, haha</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:58:29</p>
<p class="content">haha~ let go on our Monad trip</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:58:38</p>
<p class="content">unit~ join</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:58:42</p>
<p class="content">:D</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:58:45</p>
<p class="content">unit is what?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:58:54</p>
<p class="content">unit :: a -&gt; M a</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:58:59</p>
<p class="content">haha~ what is it?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:59:01</p>
<p class="content">emmm</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:59:11</p>
<p class="content">so finally it comes to return!</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:59:11</p>
<p class="content">u know it~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:59:14</p>
<p class="content">yes</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:59:18</p>
<p class="content">and the join</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:59:26</p>
<p class="content">bind</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:59:27</p>
<p class="content">fuck it ~ its not the bind</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:59:43</p>
<p class="content">join is strange</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-19 23:59:55</p>
<p class="content">join :: M (M a) -&gt; M a</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-19 23:59:59</p>
<p class="content">orz</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:00:05</p>
<p class="content">strange enough?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:00:12</p>
<p class="content">oui</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:00:22</p>
<p class="content">forget about bind first</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:00:26</p>
<p class="content">it unwraps?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:00:30</p>
<p class="content">yes</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:00:33</p>
<p class="content">unwrap~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:00:34</p>
<p class="content">haha</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:01:07</p>
<p class="content">so?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:01:06</p>
<p class="content">so with monad , these 2 super function(natual transformation)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:01:09</p>
<p class="content">we can</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:01:35</p>
<p class="content">put haskell values into a new Category</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:01:51</p>
<p class="content">wow</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:02:02</p>
<p class="content">and with join</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:02:25</p>
<p class="content">we can extract</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:02:33</p>
<p class="content">unwrap</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:02:41</p>
<p class="content">yeah</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:02:44</p>
<p class="content">now let's fuck the `bind`</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:02:51</p>
<p class="content">where is bind??</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:02:52</p>
<p class="content">where?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:02:56</p>
<p class="content">okay ...</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:03:14</p>
<p class="content">(&gt;&gt;=) :: M a -&gt; (a -&gt; M b) -&gt; M b</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:03:29</p>
<p class="content">bind :: M a -&gt; (a -&gt; M b) -&gt; M b</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:04:25</p>
<p class="content">with join comes to bind?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:04:33</p>
<p class="content">bind a f   = join (fmap f  a)</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:04:43</p>
<p class="content">WOW!</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:04:43</p>
<p class="content">think over about it, I won't tell you.</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:04:58</p>
<p class="content">i was shocked , to tell you the truth</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:05:04</p>
<p class="content">if you can calculate the result type youself~ you've know the whole thing~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:05:15</p>
<p class="content">bind a f   = join (fmap f  a)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:05:24</p>
<p class="content">remember this</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:05:37</p>
<p class="content">for Monad is just Functor ~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:05:43</p>
<p class="content">we can use fmap on it</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:05:49</p>
<p class="content">haha~~ wonderful~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:05:52</p>
<p class="content">understand?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:05:57</p>
<p class="content">per-fect</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:06:09</p>
<p class="content">so what Monad does~?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:06:34</p>
<p class="content">so what is `M`  really means</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:06:40</p>
<p class="content">haha~~~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:06:58</p>
<p class="content">just I told you Monad is endofunctor</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:07:08</p>
<p class="content">but~ have you ever noticed that</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:07:14</p>
<p class="content">it is not endo?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:07:28</p>
<p class="content">what is endo? recall it</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:07:45</p>
<p class="content">endo...</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:08:03</p>
<p class="content">map it to itself?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:08:14</p>
<p class="content">F :: M A -&gt; M A</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:09:09</p>
<p class="content">haha~ what did I said</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:09:17</p>
<p class="content">oui?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:11:10</p>

<p class="content">Feather 23:44:26
in haskell what we working on is a big category

Feather 23:44:33
we call it H

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:11:27</p>
<p class="content">the big category</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:11:47</p>
<p class="content">how big?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:11:50</p>
<p class="content">very very big</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:11:57</p>
<p class="content">any thing inside?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:12:01</p>
<p class="content">it objects is all types in Haskell</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:12:05</p>
<p class="content">all types ~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:12:14</p>
<p class="content">so :</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:12:17</p>
<p class="content">Maybe Int</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:12:25</p>
<p class="content">Maybe (Maybe Int)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:12:30</p>
<p class="content">IO Char</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:12:43</p>
<p class="content">are small categorys?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:12:51</p>
<p class="content">are just object in H</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:12:57</p>
<p class="content">because they are types</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:13:03</p>
<p class="content">think it over</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:13:09</p>
<p class="content">it is very abstract</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:13:11</p>
<p class="content">oui, i 've mistake again</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:13:19</p>
<p class="content">the big category H</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:13:28</p>
<p class="content">it objects is all types in Haskell</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:13:35</p>
<p class="content">I already told you</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:13:38</p>
<p class="content">all types~</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:13:59</p>
<p class="content">i mean , the monads like Maybe, List IO , ande etc are 'small' categories inside the big H?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:14:05</p>
<p class="content">fuck</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:14:14</p>
<p class="content">orz</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:14:16</p>
<p class="content">are objects in big H</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:14:27</p>
<p class="content">there's no small categories</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:14:34</p>
<p class="content">just a big `H`</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:14:36</p>
<p class="content">窘</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:14:41</p>
<p class="content">it objects is all types in Haskell</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:14:48</p>
<p class="content">i see</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:14:49</p>
<p class="content">what did I mean in `all`</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:14:55</p>
<p class="content">so ~ so ~</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:14:59</p>
<p class="content">the only category</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:15:01</p>
<p class="content">so endo</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:15:06</p>
<p class="content">you don't understand the bind</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:15:18</p>
<p class="content">bind a f   = join (fmap f  a)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:15:25</p>
<p class="content">what fmap does</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:15:39</p>
<p class="content">fmap f :: M a -&gt; M (M b)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:15:42</p>
<p class="content">right?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:15:58</p>
<p class="content">so M (M b) is an object here~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:16:05</p>
<p class="content">haha~</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:16:07</p>
<p class="content">en......</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:16:14</p>
<p class="content">remember the big category H</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:16:22</p>
<p class="content">and I'll show you more</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:16:35</p>
<p class="content">so tell me, what is big category H</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:16:52</p>
<p class="content">all abstract types</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:16:56</p>
<p class="content">all functions</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:17:03</p>
<p class="content">in haskell</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:17:06</p>
<p class="content">and the id</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:17:09</p>
<p class="content">right?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:17:13</p>
<p class="content">yeah</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:17:29</p>
<p class="content">haha~ not that easy</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:17:36</p>
<p class="content">haskell is very math-strict</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:17:45</p>
<p class="content">oui?</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:17:50</p>

<p class="content">so when I use
(+) :: Int -&gt; Int -&gt; Int

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:18:08</p>
<p class="content">it is an arrow form object Int to object (Int -&gt; Int)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:18:25</p>
<p class="content">so ~~ all haskell function are also objects~~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:18:32</p>
<p class="content">because they are types</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:18:36</p>
<p class="content">haha ~~~</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:18:45</p>
<p class="content">-&gt; is also type constructor</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:19:00</p>
<p class="content">if you want to understand monad~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:19:19</p>
<p class="content">so why some say  ((-&gt;) a)) is Monad~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:19:51</p>
<p class="content">understand???</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:20:15</p>
<p class="content">monad arises everywhere &gt;&lt;</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:20:28</p>
<p class="content">so ~~~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:20:30</p>
<p class="content">so ~~~~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:20:53</p>
<p class="content">big category H is horriable</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:21:47</p>
<p class="content">&gt;&lt;</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:22:10</p>

<p class="content">so why monad State can read/write/modify state~
because it transform a single type a to a function s -&gt; (s, a)

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:22:58</p>
<p class="content">a function application is a state</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:23:17</p>
<p class="content">so that when pass it a state s, it value~ that is `what the function` does changes thouhe the monad calculation</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:23:37</p>
<p class="content">so that when pass it a state s, it value~ that is `what the function does` changes though the monad calculation</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:23:59</p>
<p class="content">then at last the return value and the final state returned~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:24:22</p>
<p class="content">by this way ~ we implemented state in functional programming</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:24:26</p>
<p class="content">understand?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:24:57</p>
<p class="content">horrible</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:25:29</p>
<p class="content">mathmaticians are all talents &gt;&lt;</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:25:35</p>
<p class="content">haha~~~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:25:43</p>
<p class="content">no need to understand it</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:25:46</p>
<p class="content">just use~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:26:25</p>
<p class="content">do you understand what I told you today?</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:26:54</p>
<p class="content">not all, but much more delighted</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:27:57</p>
<p class="content">hehe~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:28:09</p>
<p class="content">enough for daily use</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:28:16</p>
<p class="content">thank you so much, never had i get know how Category comes to monad</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:28:33</p>
<p class="content">and what they are in Haskell</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:28:34</p>
<p class="content">haha~</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:28:39</p>
<p class="content"><img src="/images/Face2/0.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:28:41</p>
<p class="content">haha~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:28:47</p>
<p class="content">and at last</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:28:51</p>
<p class="content">the monad laws</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:28:57</p>
<p class="content">haha</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:29:10</p>
<p class="content">is very very simply for you now</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:29:27</p>
<p class="content">just from the definithion of category, functor, monad</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:30:01</p>
<p class="content">^_^</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:30:06</p>
<p class="content">to make sure the compostion associativity, the identity, etc. works</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:31:01</p>
<p class="content">so ~ I think you've already mastered some key points of monad</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:31:31</p>
<p class="content">haha , you are a mastered teacher!</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:32:05</p>
<p class="content"><img src="/images/Face2/20.gif" alt="" align="absmiddle" />the monad is just this simple~~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:32:17</p>
<p class="content">when you recall it</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:33:02</p>
<p class="content">Category(objects, arrows,compostion associativity,identity),</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:33:12</p>
<p class="content">Functor( A category to another )</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:33:18</p>
<p class="content">then Monad~</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:33:40</p>
<p class="content">haha , so great</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:34:12</p>
<p class="content"><img src="/images/Face2/20.gif" alt="" align="absmiddle" />聊天记录多少页了</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:34:30</p>
<p class="content">明天整理下， 哈哈</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:34:44</p>
<p class="content">好久没见中文了哇</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:34:44</p>
<p class="content"><img src="/images/Face2/20.gif" alt="" align="absmiddle" />呵呵~XXX谈话录~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:34:50</p>
<p class="content">哎~ 是哦</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:35:08</p>
<p class="content">E文都是专八的水平</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:35:16</p>
<p class="content">andelf访谈路<img src="/images/Face2/20.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:35:45</p>
<p class="content"><img src="/images/Face2/20.gif" alt="" align="absmiddle" />andelf 谈 Monad</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:36:11</p>
<p class="content">哈哈，话说某人还计划一套系列文章来</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:36:48</p>
<p class="content">注意啊~我隐去了 natual transformation 的内容, 那个没多少用, 这些就够了, 后来觉得增加复杂性, 没多大用, 以及其他一些很BT的概念</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:36:59</p>
<p class="content">呵呵~过几天再整理</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:37:04</p>
<p class="content">mark ^_^</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:37:46</p>
<p class="content"><img src="/images/Face2/20.gif" alt="" align="absmiddle" />聊天记录导出直接发表~ 还是E文版的~ 多好</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:37:59</p>
<p class="content">哈哈</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:38:05</p>
<p class="content">差不多明白就好~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:38:26</p>
<p class="content">深层理解涉及到 typed lambda calculates</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:38:33</p>
<p class="content">这个我看不懂</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:38:37</p>
<p class="content">放弃了</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:39:18</p>
<p class="content">借过一本《类型与程序设计语言》还是什么书来着，当天就还了 <img src="/images/Face2/27.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:39:30</p>
<p class="content">根本看不懂~~~哈哈~~~</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:39:43</p>
<p class="content">哈哈~~~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:39:49</p>
<p class="content">抽象代数现在计算机都不开了</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:39:54</p>
<p class="content">已经不是趋势了</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:39:55</p>
<p class="content">哎</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:40:21</p>
<p class="content">软工 &gt;&lt;</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:40:34</p>
<p class="content">因为大家要吃软饭 &gt;&lt;</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:40:44</p>
<p class="content">是啊</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:40:54</p>
<p class="content">函数式应用还是很少</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:41:03</p>
<p class="content">多的也是和命令式结合</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:41:13</p>
<p class="content">是啊</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:41:34</p>
<p class="content">.net的C#，jvm的scala</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:41:34</p>
<p class="content">纯函数式大概只有数学家才能运用自如</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:42:06</p>
<p class="content">其他语言加点FP的东西, 让会点FP的同学偶尔爽下, 来点小 hack, 这就行了~</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:42:15</p>
<p class="content">太边缘化也不好</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:42:25</p>
<p class="content"><img src="/images/Face2/20.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:42:42</p>
<p class="content">这事猪流的世界<img src="/images/Face2/20.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:42:48</p>
<p class="content">以后你再要丢人发些烂tweet我直接发短信骂你</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:43:05</p>
<p class="content">再也不敢了...窘</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:43:12</p>
<p class="content">ramen</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:43:12</p>
<p class="content"><img src="/images/Face2/20.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:43:18</p>
<p class="content"><img src="/images/Face2/34.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:43:24</p>
<p class="content"><img src="/images/Face2/0.gif" alt="" align="absmiddle" />不早了</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:43:25</p>
<p class="content">休息</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:43:34</p>
<p class="content">安 :)</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:43:47</p>
<p class="content">安~</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:44:52</p>
<p class="content">整理下谈话录发校内上</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:45:02</p>
<p class="content">^_^</p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:45:07</p>
<p class="content"><img src="/images/Face2/20.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:45:16</p>
<p class="content"><img src="/images/Face2/52.gif" alt="" align="absmiddle" /></p>

</div>
<div>
<p class="friend-id">Feather  2009-07-20 00:46:38</p>
<p class="content">休息了~明天找门新语言看</p>

</div>
<div>
<p class="my-id">Fleurer  2009-07-20 00:47:01</p>
<p class="content">哈哈 ，好梦</p>

</div>
