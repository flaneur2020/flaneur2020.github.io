use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TokenKind {
    Numeric,
    Add,
    Sub,
    Mul,
    Div,
    LParen,
    RParen,
}


#[derive(Debug, Clone, PartialEq)]
enum Token<'a> {
    Numeric(&'a str),
    Add,
    Sub,
    Mul,
    Div,
    LParen,
    RParen,
}

impl<'a> Token<'a> {
    fn kind(&self) -> TokenKind {
        match self {
            Token::Numeric(_) => TokenKind::Numeric,
            Token::Add => TokenKind::Add,
            Token::Sub => TokenKind::Sub,
            Token::Mul => TokenKind::Mul,
            Token::Div => TokenKind::Div,
            Token::LParen => TokenKind::LParen,
            Token::RParen => TokenKind::RParen,
        }
    }
}

#[derive(Clone, Debug)]
enum Expr {
    Numeric(f64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
}

#[derive(Debug)]
enum ParserError {
    EOF,
    NotImplemented,
    BadNumber(String),
    UnexpectedToken(String, String),
}

struct Tokener<'a> {
    tokens: &'a [Token<'a>],
    pos: usize,
}

impl<'a> Tokener<'a> {
    fn new(tokens: &'a [Token<'a>]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Result<&'a Token<'a>, ParserError> {
        if self.pos >= self.tokens.len() {
            return Err(ParserError::EOF);
        } 
        Ok(&self.tokens[self.pos])
    }

    fn consume<'b>(&mut self, expect: Token<'b>) -> Result<&'a Token<'a>, ParserError> {
        let got = self.peek()?;
        if got != &expect {
            return Err(ParserError::UnexpectedToken(format!("{:?}", got), format!("{:?}", expect)));
        }
        self.next()
    }

    fn next(&mut self) -> Result<&'a Token<'a>, ParserError> {
        let token = self.peek()?;
        self.pos += 1;
        Ok(token)
    }
}

trait PrefixParselet {
    fn parse_expr<'a>(&self, parser: &'a mut Parser<'_>, token: &'a Token<'_>) -> Result<Expr, ParserError>;
}

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

#[derive(Debug, Clone)]
struct InfixParselet {
    pub precedence: i32,
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

struct Parser<'a> {
    tokener: Tokener<'a>,
    prefixlets: HashMap<TokenKind, Rc<dyn PrefixParselet>>,
    infixlets: HashMap<TokenKind, InfixParselet>,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token<'a>]) -> Self {
        let mut prefixlets: HashMap<TokenKind, Rc<dyn PrefixParselet>> = HashMap::new();
        prefixlets.insert(TokenKind::Numeric, Rc::new(NumericParselet));

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

    fn parse_expr(&mut self, precedence: i32) -> Result<Expr, ParserError>{
        let token = self.tokener.next()?;
        let prefixlet = self.prefixlets
            .get(&token.kind())
            .ok_or(ParserError::UnexpectedToken(format!("{:?}", token), "Numeric".to_string()))?
            .clone();

        println!("parse_expr token: {:?}", token);
        println!("----");
        let mut left = prefixlet.parse_expr(self, token)?;
        println!("===");
        while precedence < self.get_precedence()? {
            let token = match self.tokener.next() {
                Ok(token) => token,
                Err(ParserError::EOF) => break,
                Err(e) => return Err(e),
            };
            println!("t: {:?}", token);
            let infixlet = self.infixlets
                .get(&token.kind())
                .ok_or(ParserError::UnexpectedToken(format!("{:?}", token), "Infix".to_string()))?
                .clone();
            left = infixlet.parse_expr(self, left, token)?;
        }
        println!("ok: {:?}", left);
        Ok(left)
    }

    fn get_precedence(&self) -> Result<i32, ParserError> {
        let token = match self.tokener.peek() {
            Ok(token) => token,
            Err(ParserError::EOF) => return Ok(0),
            Err(e) => return Err(e),
        };
        let precedence = self.infixlets.get(&token.kind())
            .map(|i| i.get_precedence())
            .unwrap_or(0);
        Ok(precedence)
    }
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tokens() -> Result<(), ParserError> {
        // 1 + 2 * 3 / 4 - 5
        let tokens = vec![
            Token::Numeric("1"),
            Token::Add,
            Token::Numeric("2"),
            Token::Mul,
            Token::Numeric("3"),
            Token::Div,
            Token::Numeric("4"),
            Token::Sub,
            Token::Numeric("5"),
        ];
        let mut p = Parser::new(&tokens);
        p.parse_expr(0)?;
        Ok(())
    }
}
