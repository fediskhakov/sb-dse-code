"""Towers of Hanoi — recursion practice.

Move a stack of n disks from the source peg to the target peg, one disk at a
time, never placing a larger disk on top of a smaller one.

Fill in the parts marked TODO, then run this file:

    python hanoi.py
"""


def hanoi(n, source='A', aux='B', target='C', verbose=True):
    """Move n disks from `source` to `target`, using `aux` as intermediary.

    Returns the number of moves made.

    The trick is to assume the function already works for n-1 disks, and only
    write down what to do with the bottom one:

      1. get the top n-1 disks out of the way, onto the auxiliary peg
      2. move the bottom disk from source to target
      3. put those n-1 disks back, on top of it

    Steps 1 and 3 are the same problem with one disk fewer, so each is a call
    to hanoi() itself — with the pegs playing different roles.
    """
    # TODO: base case. With no disks left there is nothing to move.
    #       Return the number of moves made in that case.

    # TODO: step 1 — move the top n-1 disks onto the auxiliary peg.
    #       Which peg is the *target* of this call? Which is the *auxiliary*?
    #       Keep the count of moves it reports.

    # step 2 — move the bottom disk across
    if verbose:
        print(f'{source} --> {target}')

    # TODO: step 3 — move the n-1 disks from the auxiliary peg onto the target.
    #       Again, work out which peg plays which role here.

    # TODO: return the total number of moves: the two recursive calls plus this one
    raise NotImplementedError('remove this line once hanoi() is written')


def check(n_max=10):
    """Check the solution against the known result: n disks take 2**n - 1 moves."""
    for n in range(1, n_max + 1):
        moves = hanoi(n, verbose=False)
        expected = 2 ** n - 1
        flag = 'ok' if moves == expected else 'WRONG'
        print(f'n = {n:2d}   moves = {moves:5d}   expected = {expected:5d}   {flag}')


if __name__ == '__main__':
    print('Sequence of moves for 3 disks:')
    hanoi(3)

    print()
    check()

    # Questions to answer once it runs:
    #  - How many moves does the solution for n disks take, and why?
    #  - What is the complexity class of this algorithm?
    #  - How long would 64 disks take, at one move per second?
