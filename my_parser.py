from cmp.pycompiler import *
from TreeDef import *
from parser_tools import *
from cmp.ast import get_printer

def DefGrammar():

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
    
    return G

    
# *******************************************************************
# ********************** Ejecutar el parser *************************
# *******************************************************************

G = DefGrammar()

equal, plus, times, minus, div, num, lpar, rpar = G.terminals

parser = LR1Parser(G, verbose=True)

tokens = [Token('5', num), Token('+', plus), Token('2', num), Token('$', G.EOF) ]
derivation = parser([tok.token_type for tok in tokens])

# Como el parser el LR(1) al trabajar sobre la forma de atributar
# la gramatica LL(1) debemos tomar los tokens y las derivaciones
# en orden inverso

tokens.reverse()
derivation.reverse()

result = evaluate_parse(derivation, tokens)

printer = get_printer(AtomicNode=ConstantNumberNode, BinaryNode=BinaryNode)
print(printer(result))