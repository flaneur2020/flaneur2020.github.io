---
date: "2022-08-16T00:00:00Z"
title: Pratt Parsing
---

递归下降是最易于上手的 Parsing 手法，容易和语法定义匹配起来，一眼望过去也比较清晰，需要魔改点 Parser 时候我这种小白也能动手。不过传统递归下降遇到表达式时就难受了：

1. 表达式有左递归的情况，每遇到左递归都需要人肉解一把，从而语法解析变得 dirty
1. 表达式需要考虑运算符优先级，手写有点麻烦

Pratt Parsing 补齐了传统递归下降的这条短板，有了它，基于递归下降也不乏工业级的 Parser 实现了。它主要解决了表达式的优先级问题，不会产生左递归，而且最多只需要 Look ahead 一个 Token 不用回溯。此外又简单又易于扩展，后悔没早学，这里拿一个基本的四则运算的例子记一把，完整代码在[这里](https://github.com/flaneur2020/flaneur2020.github.io/blob/master/exercises/pratt/src/parser.rs)。

## Parser 定义

一般意义上的 Parser 都是从 Lexer 中获取 Token 流作为输入，输出为树形结构的 AST。

拿 Parser 的结构体看一下：

``` rust
pub struct Parser<'a> {
    tokener: Tokener<'a>,
    prefixlets: HashMap<TokenKind, Rc<dyn PrefixParselet>>,
    infixlets: HashMap<TokenKind, InfixParselet>,
}
```

其中 Tokener 相当于 Lexer 的迭代器，主要有两个方法，一个是 `next(&mut self) -> Result<&'a Token<'a>, ParserError>`，能够吐回一个 Token 并移动游标；另一个是 `peek(&mut self) -> Result<&'a Token<'a>, ParserError>`  ，偷看下一个 Token，但不移动游标。

Parselet 是这里值得一提的概念，这里有 PrefixParselet 或者 InfixParselet 两种 Parselet。

## PrefixParselet

PrefixParselet 除了包括 `-1` 中的负号，也包括 `1`, `“123”` 这种字面量乃至括号，概括起来，大约是所有可以作为表达式第一个前缀的 Token，都可以由 PrefixParselet 进行处理。

PrefixParselet 的逻辑比较多样，这里定义成 trait：

``` rust
trait PrefixParselet {
    fn parse_expr<'a>(&self, parser: &'a mut Parser<'_>, token: &'a Token<'_>) -> Result<Expr, ParserError>;
}
```

NumericParselet 用于解析数值：

``` rust
struct NumericParselet;

impl PrefixParselet for NumericParselet {
    fn parse_expr<'a>(&self, _parser: &'a mut Parser<'_>, token: &'a Token<'_>) -> Result<Expr, ParserError> {
        match token {
            Token::Numeric(s) => {
                let n = s.parse::<f64>()
                    .or_else(|_| Err(ParserError::BadNumber(format!("bad num: {:?}", s))))?;
                Ok(Expr::Numeric(n))
            },
            _ => Err(ParserError::UnexpectedToken(format!("{:?}", token), "Numeric".to_string())),
        }
    }
}
```

## InfixParselet

InfixParselet 对应 + - * / 这些四则运算符，也是 Pratt Parser 处理优先级的本体之一。

在这里加减乘除处理逻辑都一样，就没有写成 trait：

``` rust
#[derive(Debug, Clone)]
struct InfixParselet {
    precedence: i32,
}

impl InfixParselet {
    fn new(precedence: i32) -> Self {
        Self { precedence }
    }

    fn parse_expr<'a>(&self, parser: &'a mut Parser<'_>, left: Expr, token: &'a Token<'_>) -> Result<Expr, ParserError> {
        let right = parser.parse_expr(self.precedence)?;
        match token.kind() {
            TokenKind::Add => Ok(Expr::Add(Box::new(left), Box::new(right))),
            TokenKind::Sub => Ok(Expr::Sub(Box::new(left), Box::new(right))),
            TokenKind::Mul => Ok(Expr::Mul(Box::new(left), Box::new(right))),
            TokenKind::Div => Ok(Expr::Div(Box::new(left), Box::new(right))),
            _ => Err(ParserError::UnexpectedToken(format!("{:?}", token), "infix operator".to_string())),
        }
    }

    fn get_precedence(&self) -> i32 {
        self.precedence
    }
}
```

每个 InfixParselet 有一个自己的优先级（precedence），比如 * / 的优先级是 20，+ - 的优先级是 10，这样乘除法的优先级就高于加减法。

注意在递归解析表达式时 InfixParselet 会将自己的 precedence 传给表达式 Parser，Parser 由此得到当前运算符的优先级，可以用于与后来的运算符做优先级比较。

## Parslet 表

所有的 Parselet 会根据它关心的首个 Token 编制到哈希表中，初始化时大约长这样：

``` rust
impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token<'a>]) -> Self {
        let mut prefixlets: HashMap<TokenKind, Rc<dyn PrefixParselet>> = HashMap::new();
        prefixlets.insert(TokenKind::Numeric, Rc::new(NumericParselet));
        prefixlets.insert(TokenKind::ParenLeft, Rc::new(ParenParselet));

        let mut infixlets = HashMap::new();
        infixlets.insert(TokenKind::Add, InfixParselet::new(10));
        infixlets.insert(TokenKind::Sub, InfixParselet::new(10));
        infixlets.insert(TokenKind::Mul, InfixParselet::new(20));
        infixlets.insert(TokenKind::Div, InfixParselet::new(20));
        Self {
            tokener: Tokener::new(tokens),
            prefixlets,
            infixlets,
        }
    }
}
```

Parselet 的抽象使 Pratt Parser 很容易扩展，增加新的语法元素，只要多一个 Parselet 定义就够了。

## Parse 逻辑本体

``` rust
pub fn parse_expr(&mut self, precedence: i32) -> Result<Expr, ParserError>{
    let token = self.tokener.next()?;
    let prefixlet = self.prefixlets
        .get(&token.kind())
        .ok_or(ParserError::UnexpectedToken(format!("{:?}", token), "Numeric".to_string()))?
        .clone();

    let mut left = prefixlet.parse_expr(self, token)?;
    while precedence < self.peek_precedence()? {
        let token = match self.tokener.next() {
            Ok(token) => token,
            Err(ParserError::EOF) => break,
            Err(e) => return Err(e),
        };
        let infixlet = self.infixlets
            .get(&token.kind())
            .ok_or(ParserError::UnexpectedToken(format!("{:?}", token), "Infix".to_string()))?
            .clone();
        left = infixlet.parse_expr(self, left, token)?;
    }
    Ok(left)
}
```

就这么多了！

parse_expr 有一个 precedence 参数，在调用时传入 0，后面会在递归中传入不同的参数表示当前运算符的优先级。

在开始解析表达式时，先拿一个 Token，查一把 PrefixParselets 的表，用 PrefixParselet 跑一把解析。

然后查 InfixParslet 表，在一个循环中持续消费 Token 并加入到 left 表达式中，直到**下一个运算符的优先级**大于**当前运算符的优先级**为止。

## 推导执行过程

代码虽然很少，行为也清晰，不过递归的过程较难在脑中直接推导，这里拿一个例子，顺着 Token 的消费来看执行的过程：

``` rust
1 + 2 + 4 * 5 - 3
```

首先是 Token “1”，它是 parse_expr(0) 解析的第一个 Prefixlet，得到表达式节点 Numeric(1.0) 保存到 left 变量中。

然后判断优先级，当前优先级（0）小于 Token “+” 的优先级（10），进入迭代。

得到 Token “+”，进入 Infixlet 解析，在 Infixlet 的 parse_expr() 中，按 precedence 为 10 调用 parser 的 parse_expr() 方法解析表达式右侧，进入 parser.parse_expr() 的二阶递归。

二阶递归中继续通过 Prefixlet 解析得到 Token “2”，随后判断当前优先级（10）等于后续优先级（10），退出迭代，返回表达式节点 Numeric(2.0) 返回第一个 Token “+” 的 parse_expr() 函数。

第一个 Token “+” 的 parse_expr() 函数将 Numeric(2.0) 加入到 left 变量中，组成表达式节点 Add(Numeric(1.0), Numeric(2.0)) 保存到 left 变量，返回 parser.parse_expr() 的一阶递归。

``` 
1  +  2  +  4  *  5  -  6
0  0  10 10
-------  ~~
|> Parser.parse_expr(0)
|  iter: 1, next token: 1
|  |> InfixParselet(+, 10).parse_expr()
|  |  |> Parser.parse_expr(10)
|  |  |  return Numeric(2.0)
|  |  return Add(Numeric(1.0), Numeric(2.0))
|  left = Add(Numeric(1.0), Numeric(2.0)), next token: +
```

这个调用图中 `----` 表示已经跑过的 Token，`|>`表示开始这个函数被调用时 Tokener 的点位，`~` 表示下一个 Token。

回到一阶递归的第二次迭代，直接画出来期间的调用栈：

``` 
1  +  2  +  4  *  5  -  6
0  0  10 0  10 10 20 10
-------------------- ~~
|> Parser.parse_expr(0) 
|  iter: 2, next token: 4
|        |> InfixParselet(+, 10).parse_expr()
|        |  |> Parser.parse_expr(10)
|        |  |  |> InfixParselet(*, 20).parse_expr()
|        |  |  |  |> Parser.parse_expr(20)
|        |  |  |  |  return Numeric(5)
|        |  |  |  return Mul(Numeric(4.0), Numeric(5.0))
|        |  |  return Mul(Numeric(4.0), Numeric(5.0))
|        |  return Add(Add(Numeric(1.0), Numeric(2.0)), Mul(Numeric(4.0), Numeric(5.0)))
|  left = Add(Add(Numeric(1.0), Numeric(2.0)), Mul(Numeric(4.0), Numeric(5.0))), next token: -
```

这里相比第一轮迭代有一个不同之处，就是跑到 Token “4” 之后，没有像第一轮迭代那样继续加入到表达式左侧，而是从 Token “4” 开始开启了新一轮表达式 Parsing，直到遇到优先级更低的 Token “-” 而退出表达式 Parsing，这时产生了 Mul(Numeric(4.0), Numeric(5.0)) 的表达式节点。优先级也就生效了。

总结来看，优先级生效的秘密，**就是每一次消费 InfixParselet 的右侧 Token 时，有一次机会判断将这个 Token 作为单值加入到左侧表达式，还是开启新的 Infix 表达式解析，也就是生成一个新的右侧 Infix 表达式**。判断的依据就是 InfixParselet 中携带的 precedence 参数，**这个参数也决定了右侧 Infix 表达式在何时退出 Parsing**，产生中间结果回到上一层递归继续 Parse。

``` 
1  +  2  +  4  *  5  -  6
0  0  10 0  10 10 20 0  10 0
-------------------------- ~
|> Parser.parse_expr(0) 
|  iter: 3, next token: EOF
|                 |> InfixParselet(-, 10).parse_expr()
|                 |  |> Parser.parse_expr(10)
|                 |  |  return Numeric(6.0)
|                 |  return Sub(Add(Add(Numeric(1.0), Numeric(2.0)), Mul(Numeric(4.0), Numeric(5.0))), Numeric(6.0))
|  left = Sub(Add(Add(Numeric(1.0), Numeric(2.0)), Mul(Numeric(4.0), Numeric(5.0))), Numeric(6.0))
|  return left
```

最后一轮迭代就比较平平无奇了，在此掠过分析过程。

到这里就得到了最终的表达式结果。

## References

- [https://journal.stuffwithstuff.com/2011/03/19/pratt-parsers-expression-parsing-made-easy](https://journal.stuffwithstuff.com/2011/03/19/pratt-parsers-expression-parsing-made-easy)
- [https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html](https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html)
- [https://www.oilshell.org/blog/2017/03/31.html](https://www.oilshell.org/blog/2017/03/31.html)
- [https://github.com/forax/pratt_parser](https://github.com/forax/pratt_parser)
- [https://github.com/munificent/bantam](https://github.com/munificent/bantam)