---
layout: post
title: "rails3中使用facebox"
tags: 
- Rails
- ruby
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

(有两个月没有更新了？这又是例行工事的凑数文。)

<!--more-->

真正接触rails的时间并不长，先前几个心不在焉的小东西烂尾之后，就一直觉得跟不上rails中各种约定的思路，也一直没有学好。直到最近在家没事搞个小项目，权当实习之前的预习，才发现上手还是很快的。所谓“约定”，就是“同一种问题使用同一种解决方法”，而且往往也正是“最好的方法”。用过一遍之后如果能留下个好印象，就该差不多了。

ajax的相关内容在Rails Guide中似乎并未提及，因此在这里记一下。假定已经实现了一个非ajax表单，就像豆瓣收藏一本书的表单一样：

<img src="http://i.min.us/iefmog.png"></img>

需要做的是，将这个表单放到<a href="http://defunkt.io/facebox/">facebox</a>里：

<img src="http://i.min.us/ibKvIy.png"></img>

<h2>jquery-ujs</h2>

先将rails默认的js框架改为jquery。

在Gemfile中加入：
<pre lang="ruby">
gem 'jquery-rails', '>= 1.0.12'</pre>

然后
<pre lang="shell">
bundle install
rails generate jquery:install</pre>

换用jquery之后，把各种js都生成到html里的link_to_remote就不能使用了。可行的做法是在link_to中加上一个:remote => true，生成的代码会像是这样：

<pre lang="html">
&lt;a href="/favorites/4e2e7a331c78b4288b000005/edit" data-remote="true"&gt; 想读 &lt;/a&gt;
</pre>

仅仅多了一个data-remote="true"，专门给jquery看的一个属性。如果用户点击这个链接，触发一个js事件弹出facebox；如果用户要从新标签中打开这个链接，就可以见到原先非ajax的那个表单。

(ps: data-remote这种自定义属性似乎是html5加入标准的？如果是&lt;a href="" remote=""&gt;&lt;/a&gt;这样写，在浏览器生成DOM时会被省略掉，jquery也就读不出来，解决方案就是加一个data-前缀。)

<h2>约定</h2>

只要是:remote => true的链接，在用户点击时就会通过ajax获取一段动态生成的js代码并执行，不同的:action对应的js代码不同。这些js代码的模板都放在views里，比如"/favorites/4e2e7a331c78b4288b000005/edit"这个RESTful的链接，对应的:controller是Favorites，:action是edit，这段js代码也就对应着edit.erb.js。

原先的Controller:

<pre lang="ruby">
class Favorites
...
  def edit
    @favorite = Favorite.find params[:id]
    @favorite.state = params[:state] if params[:state]
  end
 ...
 end</pre>
 
修改后：
<pre lang="ruby">
class Favorites
...
  def edit
    @favorite = Favorite.find params[:id]
    @favorite.state = params[:state] if params[:state]
    respond_to do |format|
      format.html 
      format.js { render :ajax_edit, :layout => false }
    end
  end
..
end</pre>

其中respond_to是根据mime类型分派render的内容。获取的是js代码，mime类型肯定就是text/javascript了。在这段js里面打开facebox即可。

表单的内容呢？嵌到js代码里面。重构下代码，把edit.haml中的内容移动到_form_edit.haml中，在edit.haml里只留一行=render :template => 'favorites/_form_edit.haml'

ajax_edit.erb.js
<pre lang="js">
var form_html = '<%= escape_javascript(render :template => 'favorites/_form_edit') %>';
$.facebox(form_html);
</pre>

javascript并无heredoc那种多行字符串的语法，escape_javascript可以将多行的字符串转义成单行。

这样就好了。提交表单时还会刷新下页面，不过若要修改成不需刷新的ajax表单，也就是按这路数再走一遍的功夫。
