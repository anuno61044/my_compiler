from cmp.pycompiler import *
from TreeDef import *
from parser_tools import *
from AtributedGrammar import *

G = Grammar()

# No terminales
E = G.NonTerminal('E', True)
A,T,F = G.NonTerminals('A T F')

# Terminales
equal, plus, times, minus, div, num, lpar, rpar = G.Terminals('= + * - / float ( )')

# Reglas
E %= A, lambda h,s : s[1]

A %=  A + plus + T, lambda h,s: PlusNode(s[1],s[3])
A %=  A + minus + T, lambda h,s: MinusNode(s[1],s[3])
A %= T, lambda h,s : s[1]

T %= T + times + F, lambda h,s: StarNode(s[1],s[3])
T %= T + div + F, lambda h,s: DivNode(s[1],s[3])
T %= F, lambda h,s : s[1]

F %= num, lambda h,s : ConstantNumberNode(s[1])
F %= lpar + A + rpar, lambda h,s : s[2]


parser = LR1Parser(G, verbose=True)

tokens = [Token('5', num), Token('*', times), Token('(', lpar), Token('2', num), Token('+', plus), Token('4', num), Token(')', rpar), Token('/', div), Token('411', num), Token('$', G.EOF) ]
derivation = parser([tok.token_type for tok in tokens])

tokens.reverse()
derivation.reverse()

print('\n\n',derivation)
print(tokens)
print('\n')

result = evaluate_parse(derivation, tokens)
from cmp.ast import get_printer
printer = get_printer(AtomicNode=ConstantNumberNode, BinaryNode=BinaryNode)
print(printer(result))

