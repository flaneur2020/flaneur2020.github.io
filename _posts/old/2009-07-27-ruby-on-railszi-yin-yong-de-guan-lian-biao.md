---
layout: post
title: "ruby on rails:自引用的关联表"
tags: 
- Rails
- ruby
- trick
- "翻译"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

原文：<a href="http://dizzy.co.uk/ruby_on_rails/contents/self-referential-table-joins">http://dizzy.co.uk/ruby_on_rails/contents/self-referential-table-joins</a>
翻译：ssword

本文将展示如何为表创建与自身的关联。例如，一篇文章可以有多个相关文章，抑或一个人可以认识很多人。

设想，你有个表存放一个blog中的所有文章，每篇文章都可以有几篇相似的文章。或者有个储存人物资料的表，人与人之间也可能存在相互的关系。不难看出这是多对多（has_and_belongs_to_many，HABTM）的关联，不过只有一个表。因此这是个自引用的关联：一个人可以认识多个人，这些人还可以认识别的人，等等。

同其他多对多的关联一样，我们需要另加一个中间表，来记录文章之间的关系。只是这里中间表的两个id都是来自article，一个字段记录主文章的id，另一个字段表示相关文章的id。把这个中间表命名为 related_articles，创建一个migration...

<pre lang="ruby">
create_table "related_articles", :force => true, :id => false do |t|
    t.column "related_article_id", :integer
    t.column "main_article_id", :integer
end
</pre>

<strong>自引用类
</strong>

回想下，中间表不另必拥有自己的id，因此我们设置:id => false。再看下Article类：

除却在创建关联时添加的冗余内容，Article与其他的类并无二致。在标准的多对多关联中，Rails会自动找到中间表的名字。不过在这里，中间表的名字并非两个Model的连接，它是自引用。因此我们需要添加点额外选项，让它辨认出来：

<pre lang="ruby" line="1">
class Article < ActiveRecord::Base
    has_and_belongs_to_many :related_articles, :class_name => "Article", :join_table => "related_articles", :foreign_key => "main_article_id", :association_foreign_key => "related_article_id"
end
</pre>

这几个选项又是什么意思呢？
<ul>

    <li>:join_table向rails指明了中间表的名字，也就是related_articles。</li>
    <li>:foreign_key指明了外键的名字，它代表主文章的id，作为相关文章的上一级：称作main_article_id。</li>
    <li>:association_foreign_key指明了相关文章的外键，也就是"属于"主文章的文章。</li>
    <li>:class_name表示了当前类的名字，即Article </li>

</ul>

<strong>实现</strong>

添加相关文章的界面最好是下拉列表。修改view new.rhtml，让我们可以在添加新文章时添加相关文章。

<pre lang="ruby" line="1">
<% form_for(@article) do |f| %>

  <b>Content</b><br />
  <%= f.text_field :content %>

  <b>Title</b><br />
  <%= f.text_field :title %>

  <b>Related articles</b><br />
  <%= f.collection_select(:related_article_ids, Article.find(:all), :id, :title, {}, :multiple => true) %>

  <%= f.submit "Create" %>

<% end %>
</pre>

我们用了helper函数collection_select，结合related_article_ids方法。在定义多对多关联的时候，这个方法就自动生成了，它接受一个id构成的数组来添加到中间表。通过Article.find(:all)获得数据库中的所有文章，交给collection_select格式化输出，这一来在下拉列表中显示:title，而在提交表单的时候就得到了所选的文章:id。同时设置:multiple => true，从而允许多选。

我不讲太多细节了，像如何在多对多关联时使用下拉列表之类，那是另一个话题。不过别担心 - 我很快就会写（看看右边的相关文章，或许在你看到的时候我已经写了!） （译者注：以<a href="http://dizzy.co.uk/ruby_on_rails/contents/self-referential-table-joins">作者的blog</a>为准）
