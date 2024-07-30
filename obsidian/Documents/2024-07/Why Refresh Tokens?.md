> There is a scalability reason, in that the access_token could be verifiable on the resource server without DB lookup or a call out to a central server, then the refresh token serves as the means for revoking in the "an access token good for an hour, with a refresh token good for a year or good-till-revoked."
>
> There is a security reason, the refresh_token is only ever exchanged with authorization server whereas the access_token is exchanged with resource servers.  This mitigates the risk of a long-lived access_token leaking (query param in a log file on an insecure resource server, beta or poorly coded resource server app, JS SDK client on a non https site that puts the access_token in a cookie, etc) in the "an access token good for an hour, with a refresh token good for a year or good-till-revoked" vs "an access token good-till-revoked without a refresh token."

> On Jun 15, 2011, at 11:56 AM, Eran Hammer-Lahav wrote:

> > Yes, this is useful and on my list of changes to apply.
> > 
> > But I would like to start with a more basic, normative definition of what a refresh token is for. Right now, we have a very vague definition for it, and it is not clear how developers should use it alongside access tokens.
> > 
> > EHL

