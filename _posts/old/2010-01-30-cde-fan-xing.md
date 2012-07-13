---
layout: post
title: "C的泛型库khash"
tags: 
- C
- trick
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

写小东西需要用到哈希表这样的数据结构时候才后悔没用C++，最起码还有stl能用来着。虽说C也能搞泛型，不过宏实现的泛型...真的好恐怖。

tinyrb用了khash做哈希表，据作者说已经是一个稳定的实现了。

khash.h的内容：
<a href="http://attractivechaos.awardspace.com/khash.h.html">http://attractivechaos.awardspace.com/khash.h.html</a>

大体可以这么用：
<pre lang="c">
#include "stdio.h"
#include "khash.h"
KHASH_MAP_INIT_STR(str, int) //以“str”这名字初始化一个类型的map，键类型为字符串，值类型为int
int main() {
    int ret, is_missing;
    khiter_t k; //khash的索引器，好像就是个int
    khash_t(str) *h = kh_init(str); //str只是个名字，初始化
    k = kh_put(str, h, "test", &ret); //“test”即键，ret判定操作是否成功，返回k为索引器

    if (!ret) kh_del(str, h, k); //如果h中已经存在了这个键，就删除之
    kh_value(h, k) = 10; //设置键“test”对应的值（10）
    
    printf("%d\n", kh_val(h,k)); //kh_val(h,k)即10

    k = kh_get(str, h, "test"); //获得“test”对应的索引器k
    printf("%d\n", kh_val(h,k)); //得10

    kh_destroy(str, h);
    return 0;
}
</pre>

终究不如模板来的自然，呵呵~不过也不错了。
