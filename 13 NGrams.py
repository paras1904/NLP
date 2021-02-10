# vid 14
def N_grams(text,n):
    c = str(text)
    token = c.split()
    nagrams = []
    for i in range(len(token)-n+1):
        temp = [token[j] for j in range(i,i+n)]
        nagrams.append(''.join(temp))
    return nagrams
text = 'the quick brown fox jump over the lazy dog'
n = 3
for i in N_grams(text,n):
    print(i)