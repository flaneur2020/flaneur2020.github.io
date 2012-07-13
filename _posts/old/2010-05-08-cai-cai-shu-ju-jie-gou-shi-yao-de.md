---
layout: post
title: "踩踩数据结构什么的 =“="
tags: 
- haskell
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---


先从一棵二叉查找树开始好了。挖坑慢慢填，时间有的是(?)

<pre lang="haskell">
module BTree where

data BTree a b = Empty 
               | BNode {
                    kv :: (a, b), 
                    left :: BTree a b,
                    right :: BTree a b
               } 
               deriving(Show, Eq)

insert :: (Ord a) => (a, b) -> BTree a b -> BTree a b
insert (k,v) Empty = BNode (k, v) Empty Empty
insert (k,v) pnode@(BNode (pk, _) lnode rnode) 
    | k == pk   = pnode { kv = (k,v) }
    | k <  pk   = pnode { left  = insert (k,v) lnode }
    | k >  pk   = pnode { right = insert (k,v) rnode }

find :: (Ord a) => a -> BTree a b -> Maybe b
find k Empty = Nothing
find k (BNode (pk,pv) lnode rnode)
    | k == pk = Just pv
    | k <= pk = find k lnode
    | k >  pk = find k rnode
find _ _ = Nothing

remove :: (Ord a) => a -> BTree a b -> BTree a b
remove k Empty = Empty
remove k pnode@(BNode (pk,pv) lnode rnode) 
    | k == pk = merge lnode rnode
    | k <= pk = pnode { left = remove k lnode }
    | k >  pk = pnode { right = remove k rnode }

merge :: (Ord a) => BTree a b -> BTree a b -> BTree a b
merge lnode rnode = fromList $ (toList lnode) ++ (toList rnode)

--helper
isEmpty Empty = True 
isEmpty _     = False

fromList :: (Ord a) => [(a,b)] -> BTree a b
fromList = foldl (flip insert) Empty 

toList :: BTree a b -> [(a,b)]
toList Empty = [] 
toList (BNode (k,v) lnode rnode) = 
    (toList lnode) ++ [(k,v)] ++ (toList rnode)

-- for test
root = fromList $ [
    (4, "fleurer"), 
    (10, "ssword"), 
    (100, "ssword"), 
    (2, "xx")]

</pre>

haskell做数据结构好像别扭的很...不用ST Monad或者IO的话什么都是值还不能引用，像二叉树的旋转什么的就没想出常数时间的办法。像上面那个remove就直接简单粗暴了orz

(邪恶音：用ST Monad不就好了嘛)
(ST Monad都用了还用haskell干嘛 TvT)
