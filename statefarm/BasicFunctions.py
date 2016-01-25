

friends = ['john', 'pat', 'gary', 'michael']
for i, name in enumerate(friends):
    print "name %i is %i" % (i, name)

# How many friends contain the letter 'a' ?
count_a = 0
for name in friends:
    if a in name:
        count_a++

print "%f percent of the names contain an 'a'" % ( count_a / len(friends) )


# Say hi to all friends
def print_hi(greeting='hello', name):
    print "%s %s" % (greeting, name)

map(print_hi, friends)

# Print sorted names out
print friends.sort()


/*
    Calculate the factorial N! = N * (N-1) * (N-2) * ...
*/

def factorial(x):
    """
    Calculate factorial of number
    :param N: Number to use
    :return: x!
    """
    if x==1: return 1
    return factorial(x-1)

print "The value of 5! is", factorial(5)
