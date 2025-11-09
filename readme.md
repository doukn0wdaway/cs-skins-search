Tbh, I'll stick with [fuzzysort](https://www.npmjs.com/package/fuzzysort); it seems more convenient
Yes, the search takes about 5-10ms for ~30k entries, but it works in a more conventional way (thing like "printstream ft" work on it, for example)

The other two (MiniSearch and search using cosine similarity embeddings) work:

- the first one doesn't handle short words in the search query properly, like "printstream ft" - it just doesn't react to the letters "ft" (but it's really fast).
- As for the second one: I don't want to trade functionality for search speed (on my machine, the search was around 50ms). Also, I don't need something like search for similar words - I just need pure fuzzy search.

[data source](https://bymykel.com/CSGO-API/#list-skins)
