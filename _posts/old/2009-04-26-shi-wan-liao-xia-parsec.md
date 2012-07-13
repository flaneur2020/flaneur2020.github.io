---
layout: post
title: "试玩了下parsec"
tags: 
- FP
- haskell
- parsec
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

上个星期翻的那篇文章里貌似提了下parsec，照着文档试玩了下。之前貌似玩过ruby的那个代码生成工具racc，它的名气不是很大而且文档不怎么全，感觉不是很爽。写了个helloworld就再也没碰。再写个parsec的helloworld。

语法分析的helloword就是表达式计算了。parsec跟*acc貌似不怎么像。不过无非都是上下文无关文法嘛，这个有名的ebnf：
<pre lang="pascal">
expr     ::=   expr  ’+’   term  |  term
term     ::=   term  ’*’   factor   |  factor
factor   ::=   ’(’  expr   ’)’  |  digit+

digit    ::=   ’0’  |  ’1’  |  ...  |  ’9’
</pre>
葫芦画瓢地写出parsec版：
<pre lang="haskell">
import Text.ParserCombinators.Parsec
import qualified Text.ParserCombinators.Parsec.Token as P
import Text.ParserCombinators.Parsec.Language (haskellDef)

do_parse p input
	= case (parse p "" input) of
		Left err -> do {
			putStr "parse error at:";
			print err;
		}
		Right x -> do {
			print x;
		}

expr :: Parser Int
expr = do {
	a < - expr;
	char '+';
	b <- term;
	return (a + b);
} <|> term

term :: Parser Int
term = do {
	a < - term;
	char '*';
	b <- factor;
	return (a*b);
}

factor :: Parser Int
factor = do {
	char '(';
	a <- expr;
	char ')';
	return a;
} <|> number

number = do {
	a < - many1 digit;
	return (read a);
}
</pre>
parsec， Parser Combinators嘛。当然跟*acc那套不一样了。函数在这里就都是个组合子，可以像零件那样方便地组合。

do_parse这个函数取两个参数，一个是parser，一个是表示表达式的字符串。haskell的do-notation除了允许像python那样的缩进风格之外，还有这里用的c-style。parsec就用monad来表示sequence，< |>来表示choice，某种意义上讲，貌似只要它俩就足够构造出复杂的parser了。咋一看可能要比*acc那伪bnf的观感麻烦得多，但别忘了组合子的优势：可以方便地组合。用几个简单的组合子可以方便地构造出复杂的组合子，复杂的组合子继续组合成更复杂的组合子，而使用起来可是极为简洁。

嗯，废话真多。
进ghci测试一下
</pre><pre lang="haskell">
*Main> do_parse expr "1+1"
*** Exception: stack overflow
</pre>
查手册，发现这么一句“Unfortunately, left-recursive grammars can not be specified directly in a combinator library. If you accidently write a left recursive program, the parser will go into an infinite loop! ”

哑然。
组合子还是函数嘛，想想在c中要是定义这样的函数会怎样。
<pre lang="c">
Paser expr(){
	a = expr();
	...
}
</pre>
不过“However, every left-recursive grammar can be rewritten to a non-left- recursive grammar.  The library provides combinators which do this automatically for you (chainl and chainl1). ”

嗯。手册里貌似写了个表达式计算的例子，就直接用了Parsec内置的ParsecExpr：
<pre lang="haskell">
import   ParsecExpr

expr      ::  Parser   Integer
expr      =  buildExpressionParser      table   factor
          < ?>  "expression"

table     =  [[op  "*"  (*)  AssocLeft ,   op  "/"  div  AssocLeft]
             ,[op  "+"  (+)  AssocLeft,    op  "-"  (-)  AssocLeft]
             ]
          where
             op  s f  assoc
                =  Infix   (do{  string   s;  return  f})   assoc

factor    =  do{  char  ’(’
                ; x  < - expr
                ; char  ’)’
                ; return   x
               }
          <|>  number
          < ?>  "simple   expression"

number    ::  Parser   Integer
number    =  do{  ds < -  many1   digit
                ; return   (read   ds)
               }
          <?>  "number"
</pre>
呃，运算符的优先级，左结合都考虑了。不过这样也太没意思了，该我写的都让它给内置了，不过那句话不是提到有个chainl1么，就用chainl1重写：
<pre lang="haskell">
module Main where

import Text.ParserCombinators.Parsec
import qualified Text.ParserCombinators.Parsec.Token as P
import Text.ParserCombinators.Parsec.Language (haskellDef)

main = do {
	input < - getLine;
	do_parse expr input;
}

do_parse p input
	= case (parse p "" input) of
		Left err -> do {
			putStr "parse error at:";
			print err;
		}
		Right x -> do {
			print x;
		}

lexer = P.makeTokenParser haskellDef
parens = P.parens lexer
symbol = P.symbol lexer
naturalOrFloat = P.naturalOrFloat lexer

expr = term `chainl1` addop
term = factor `chainl1` mulop
factor = parens expr < |> number
number = do {
	norf < - naturalOrFloat;
   	case norf of
		Left n -> return (read $ show n);  --so sucks
	   	Right f	-> return f;
}

mulop = do { symbol "*" ; return (*); } < |> do { symbol "/" ; return (/); }
addop = do { symbol "+" ; return (+); } < |> do { symbol "-" ; return (-); }
</pre>
用到了ParsecToken。它就直接使用了haskell的token定义，也就是说，你可以在这个程序里使用haskell的注释
<pre lang="haskell">
*Main> do_parse expr "1+{-- it's a comment --}(1/2)"
1.5
</pre>
