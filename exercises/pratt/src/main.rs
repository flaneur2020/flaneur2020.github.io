use collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
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

enum Expr {
    Numeric(f64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
}

enum ParserError {
    EOF,
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
        if self.pos > self.tokens.len() {
            return Err(ParserError::EOF);
        } 
        Ok(&self.tokens[self.pos])
    }

    fn consume<'b>(mut self, expect: Token<'b>) -> Result<&'a Token<'a>, ParserError> {
        let got = self.peek()?;
        if got != &expect {
            return Err(ParserError::UnexpectedToken(format!("{:?}", got), format!("{:?}", expect)));
        }
        self.next()
    }

    fn next(mut self) -> Result<&'a Token<'a>, ParserError> {
        let token = self.peek()?;
        self.pos += 1;
        Ok(token)
    }
}

trait Parserlet {
    fn parse_expr<'a>(&self, tokener: &'a Tokener<'a>) -> Result<Expr, ParserError>;
}

struct InfixParserlet {
    pub precedence: i32,
    pub is_right_assoc: bool,
}

impl InfixParserlet {
    fn new(precedence: i32, is_right_assoc: bool) -> Self {
        Self { precedence, is_right_assoc }
    }
}

struct Parser<'a> {
    tokener: Tokener<'a>,
    infixlets: HashMap<TokenKind, InfixParserlet>,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token<'a>]) -> Self {
        let infixlets = HashMap::new();
        infixlets.insert(TokenKind::Add, InfixParserlet::new(10, false));
        infixlets.insert(TokenKind::Sub, InfixParserlet::new(10, false));
        infixlets.insert(TokenKind::Mul, InfixParserlet::new(20, false));
        infixlets.insert(TokenKind::Div, InfixParserlet::new(20, false));
        Self {
            tokener: Tokener::new(tokens),
            infixlets,
        }
    }

    fn parse(mut self, precedence: i32) -> Result<Expr, ParserError>{
        let token = self.tokener.next();
    }
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tokens() {
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
    }
}
