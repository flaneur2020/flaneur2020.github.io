---
layout: post
title: "Quick Note: Radix Tree in Ngnix"
---

强扣教科书的字眼的话，nginx中的radix tree似乎更像bitwise trie。bitwise trie的性质不难理解：

1. 从key的二进制最高位起，按位区分左子树还是右子树
2. 数据只保存于叶节点
3. 因此:
   + 特定key的查找路径总是固定的
   + 树的高度等于key的位数

nginx中默认的模块中，只有 `http/modules/ngx_http_geo_module.c` 用到了 `radix_tree` 。它可以依据ip的范围划定出地理区域。

这里记一下nginx的实现中有趣的地方，为了与教科书上的Radix Tree区分，这里统称为 `radix_tree`。

---------------------

## Code Notes

```
ngx_radix_tree_create(ngx_pool_t *pool, ngx_int_t preallocate)
```

关于`preallocation`参数有一大段注释：

```
/*
* Preallocation of first nodes : 0, 1, 00, 01, 10, 11, 000, 001, etc.
* increases TLB hits even if for first lookup iterations.
* On 32-bit platforms the 7 preallocated bits takes continuous 4K,
* 8 - 8K, 9 - 16K, etc. On 64-bit platforms the 6 preallocated bits
* takes continuous 4K, 7 - 8K, 8 - 16K, etc. There is no sense to
* to preallocate more than one page, because further preallocation
* distributes the only bit per page. Instead, a random insertion
* may distribute several bits per page.
*
* Thus, by default we preallocate maximum
* 6 bits on amd64 (64-bit platform and 4K pages)
* 7 bits on i386 (32-bit platform and 4K pages)
* 7 bits on sparc64 in 64-bit mode (8K pages)
* 8 bits on sparc64 in 32-bit mode (8K pages)
*/

```

提升TLB命中率的优化措施，意在减少算法过程中踩过的页面数量，尽量使内存访问局部化到一个页面以内。根节点必然会被访问，那么顺便将根部附近的几个节点安排到同一个页面是合理的。

不过 `There is no sense to to preallocate more than one page， because further preallocation distributes the only bit per page.`<sup>1</sup>，
再往下的子节点，如果超出一个页面的界限，就没有必要在这里预分配了。
因为特定key的查找路径总是固定的性质，将这段查找路径中经过的节点安排到同一个页面无疑更加合理。
局部化策略并非试图将所有的数据访问局部化，而是将不相关的数据分散开。

`"the only bit per page"` 这句里的 "bit" 单词有点奇怪，考虑到 `radix_tree` 的层数与key的位数相等的性质，我想可以把它当作 "层" 来理解。

假如有这么一棵 `radix_tree`:


```

               root
             /        \
           0            1
         /   \        /   \
        00   01      10    11
      /   \   
   000
```    

往里面插入一条key为00110的数据时，会成为这样子:


```
               root
             /        \
           0           1
         /   \        /  \
       00     01     10   11
      /   \
   000    001 *
             \      
             0011 *
            /
        00110 *
```

日后若查找 00110，则 001, 0011, 00110 这几个节点将必然经过，将它们安排在一个页面(至少是临近的内存区域)将更加合理。

```
ngx_radix32tree_insert(ngx_radix_tree_t *tree, uint32_t key, uint32_t mask, uintptr_t value)
```

理论上 `radix_tree 的深度等于key的位数，不过`key`是 `int32_t` 类型总是32位。这里通过 `mask` 参数来限制trie的深度。比如一开始预分配时只希望有6层，就将 `mask` 设为 `0b111111000000…`这样子。

nginx有将子网掩码直接作为这里的 `mask`，不过`mask`并不一定非子网掩码不可。

```
ngx_radix32tree_delete(ngx_radix_tree_t *tree, uint32_t key, uint32_t mask)
```

+ radix_tree 的数据仅存在于叶子节点，在删除叶子节点时，作为路径的中间节点也会被清理。清理的条件:
  1. 没有子节点
  2. 自身的值为空(`NGX_RADIX_NO_VALUE`)
  3. 并非根节点
+ 被清理的节点会被nginx放到freelist里面，方便下次利用:

```
node->right = tree->free;
tree->free = node;
```

这个freelist的构造也蛮有趣，顺着节点的right字段链起来。

## Footnotes

1. 这里一个typo哪位同学无聊可以补下 :D

## Reference
+ http://www.cs.waikato.ac.nz/Teaching/COMP317B/Week_4/PATRICIA.html
+ http://www.cs.waikato.ac.nz/Teaching/COMP317B/Week_4/digital_search_tree.html
+ [HttpGeoModule](http://wiki.nginx.org/HttpGeoModule)
+ [Design and Analysis of Algorithms 2006](http://www.cs.waikato.ac.nz/Teaching/COMP317B/2006index.html)


