
> A good design lends itself to efficient testing—striving for a good test suite helps arrive at a good design. This is not to say that the code should be adjusted to the tests. And is not to say that the tests should be driving the implementation.
> 
> This interrelation between design and testing is best illustrated in chapter 7, where the author suggests an ideal structure for the codebase, and shows how to refactor code towards that structure, thus enabling effective tests.

![[Pasted image 20250802222107.png]]

> Overcomplicated code should be split into _deep_ domain classes, to be thoroughly unit tested, and _wide_ controllers, exercised by strategic integration tests.



![[Pasted image 20250802222154.png]]
> I found this notion interestingly similar to the discussion of _module depth_ from _A Philosophy of Software Design_:
![[Pasted image 20250802222207.png]]### Chapter 

## 7: Refactoring toward valuable unit tests

- All production code can be categorized along two dimensions:
    - Complexity or domain significance.
    - The number of collaborators.
- This categorization gives us four kinds of code:
    - **Trivial code** (low complexity/significance, few collaborators): this code shouldn’t be tested at all
    - **Domain model and algorithms** (high complexity/significance, few collaborators): this code should be unit tested. The resulting unit tests are highly valuable and cheap.
    - **Controllers** (low complexity/significance, many collaborators): controllers should be briefly tested as part of overarching integration tests.
    - **Overcomplicated code** (high complexity/significance, many collaborators): this code is hard to test, and as such it’s better to split it into domain/algorithms and controllers.
- The domain model encapsulates the business logic and the controllers deal with the orchestration of collaborators. You can think of these two responsibilities in terms of _code depth_ versus _code width_. Your code can be either deep (complex or important) or wide (work with many collaborators), but not both.
- Getting rid of the overcomplicated code and unit testing only the domain model and algorithms is the path to a highly valuable, easily maintainable test suite. With this approach, you won’t have 100% test coverage, but you don’t need to.