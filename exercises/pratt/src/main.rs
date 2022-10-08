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

struct Parser<'a> {
    tokens: &'a [Token<'a>],
    pos: usize,
}

impl<'a> Parser<'a> {
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
