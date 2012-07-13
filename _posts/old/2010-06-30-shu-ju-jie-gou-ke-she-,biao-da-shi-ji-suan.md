---
layout: post
title: "数据结构课设，表达式计算"
tags: 
- Algorithm
- C
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

随便写了下，没仔细调试。

大约这么定义
double parse_expr(char *input, char **output);

input就是表达式的字符串，output是解析剩下的部分。如果output==input，就是解析失败。如果正确，返回运算结果。

用起来大约这样
<pre lang="c">
    char *out, *exp="2+(1+3)*4";
    double r = parse_expr(exp, &out);
    if (out==exp){
        printf("parse fail\n");
        return 0;
    }
    printf("%s => %f \n", exp, r);
</pre>

回头想下，思路跟<a href="http://github.com/Fleurer/FParser">以前写的parser</a>几乎是葫芦画瓢，也该算是递归下降吧。

代码如下

exp.h
<pre lang="c">
double parse_expr(char *input, char **output);
double parse_term(char *input, char **output);
double parse_fact(char *input, char **output);
double parse_numb(char *input, char **output);
</pre>
exp.c
<pre lang="c">
#include "exp.h"

// expr ::= term {[+-] term}
double parse_expr(char *input, char **output){
    double r=0;
    char *tmp, op;
    tmp = input;
    // factor
    r = parse_term(tmp, output);
    if (output==tmp){
        return 0;
    }
    // {[+-] term}
    tmp = *output;
    while ((op=*tmp)=='+' || op=='-'){
        double r2 = parse_term(++tmp, output);
        if (tmp==*output){
            *output = input;
            return 0;
        }
        r = (op=='+')? r+r2: r-r2;
        tmp = *output;
    }
    return r;
}

// term ::= factor {[*/] factor}
double parse_term(char *input, char **output){
    double r=0;
    char *tmp, op;
    tmp = input;
    // factor
    r = parse_fact(tmp, output);
    if (output==tmp){
        return 0;
    }
    // {[*/] factor}
    tmp = *output;
    while ((op=*tmp)=='*' || op=='/'){
        double r2 = parse_fact(++tmp, output);
        if (tmp==*output){
            *output = input;
            return 0;
        }
        r = (op=='*')? r*r2: r/r2;
        tmp = *output;
    }
    return r;
}

// fact ::= (expr) | numb
double parse_fact(char *input, char **output){
    double r=0;
    char *tmp;
    // (expr)
    if (*input=='(') {
        tmp = input+1;
        r = parse_expr(tmp, output);
        if(**output==')'){
            (*output)++;
            return r;
        }
        // if fail, output == input
        else {
            *output = input;
            return 0;
        }
    }
    // numb
    r = parse_numb(input, output);
    return r;
}

double parse_numb(char *input, char **output){
    double  r = 0;
    char ch, *tmp=input;
    while((ch = *tmp++)>='0' && ch<='9'){
        r = r*10+ch-'0';
    }
    *output = tmp-1;
    return r;
}

</pre>

main.c
<pre lang="c">
int main(){
    double r;

    char *out, exp[1024];
    scanf("%256s", exp);
    r = parse_expr(exp, &out);
    if (out==exp){
        printf("parse fail\n");
        return 0;
    }
    printf("%s => %f \n", exp, r);

    return 0;
}
</pre>
