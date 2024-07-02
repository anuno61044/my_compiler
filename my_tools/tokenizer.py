from cmp.pycompiler import Grammar
from my_tools.TreeRegex import *
from cmp.utils import Token
from my_tools.parser_tools import *

class RegexHandler:
    def __init__(self):
        self.grammar = self._build_grammar()
    
    def _build_grammar(self):
        G = Grammar()
        E = G.NonTerminal('E', True)
        T, F = G.NonTerminals('T F')
        self.pipe, self.star, self.opar, self.cpar, self.symbol = G.Terminals('| * ( ) symbol')

        # > PRODUCTIONS??? LR(1) Grammar
        E %= E + self.pipe + T, lambda h,s: UnionNode(s[1],s[3])
        E %= T, lambda h,s: s[1]

        T %= T + F, lambda h,s: ConcatNode(s[1],s[2])
        T %= F, lambda h,s: s[1]

        F %= self.opar + E + self.cpar, lambda h,s: s[2]
        F %= F + self.star, lambda h,s: ClosureNode(s[1])
        F %= self.symbol, lambda h,s: SymbolNode(s[1])

        return G

    def _regex_tokenizer(self, text, skip_whitespaces=True):
        tokens = []
        tmp = ''
        text = text + '$'

        for char in text:
            if skip_whitespaces and char.isspace():
                continue
            if char == '|':
                if len(tmp) > 0: 
                    for t in tmp:
                        tokens.append(Token(t,self.symbol))
                tokens.append(Token(char,self.pipe))
                tmp = ''
            elif char == '*':
                if len(tmp) > 0:
                    for t in tmp:
                        tokens.append(Token(t,self.symbol))
                tokens.append(Token(char,self.star))
                tmp = ''
            elif char == '(':
                if len(tmp) > 0:
                    for t in tmp:
                        tokens.append(Token(t,self.symbol))
                tokens.append(Token(char,self.opar))
                tmp = ''
            elif char == ')':
                if len(tmp) > 0:
                    for t in tmp:
                        tokens.append(Token(t,self.symbol))
                tokens.append(Token(char,self.cpar))
                tmp = ''
            elif char == '$':
                if len(tmp) > 0:
                    for t in tmp:
                        tokens.append(Token(t,self.symbol))
                tokens.append(Token('$', self.grammar.EOF))
                break
            else:
                tmp = tmp + char
            
        return tokens

    def _build_automaton(self, text):

        # Obtener los tokens de la expresion regular
        tokens = self._regex_tokenizer(text)
        # print(tokens,'\n')

        parser = LR1Parser(self.grammar)
        derivations = parser([tok.token_type for tok in tokens])
        # print(derivations,'\n')

        tokens.reverse()
        derivations.reverse()

        result = evaluate_parse(derivations, tokens)

        nfa = result.evaluate()

        dfa = nfa_to_dfa(nfa)

        return dfa

    def __call__(self,text):
        return self._build_automaton(text)

# ************************************************
# ******************** TEST **********************
# ************************************************

def getAceptedTag(regexs):
    max = 0
    accepted_regex = None

    for r in regexs:
        if r['count'] > max:
            max = r['count']
            accepted_regex = r
    
    if max == 0:
        return None

    last_state = accepted_regex['state']
    autom = accepted_regex['autom']

    if last_state.issubset(autom.finals):
        return accepted_regex['tag']
    else:
        return None



if __name__ == "__main__":

    nonzero_digits = '|'.join(str(n) for n in range(1,10))
    letters = '|'.join(chr(n) for n in range(ord('a'),ord('z')+1))

    automMaker = RegexHandler()

    dfa1 = automMaker('for')
    dfa2 = automMaker(f'({nonzero_digits})(0|{nonzero_digits})*')
    dfa3 = automMaker(f'({letters})({letters}|0|{nonzero_digits})*')

    text = 'fore324 23fd for'

    while text:
        regexs = [
            {
                "tag": "num",
                "autom":dfa2,
                "active":True,
                "state":[0],
                "count":0
            },
            {
                "tag": "for",
                "autom":dfa1,
                "active":True,
                "state":[0],
                "count":0
            },
            {
                "tag": "ident",
                "autom":dfa3,
                "active":True,
                "state":[0],
                "count":0
            }
        ]

        count = 0

        for c in text:

            count += 1

            if c == ' ':
                break

            for regex in regexs:
                if not regex['active']:
                    continue

                state = regex['state']
                new_state = move(regex['autom'],state,c)
                
                if new_state == set():
                    regex['active'] = False
                    continue

                regex['count'] += 1
                regex['state'] = new_state
            
            accepted = False
            for r in regexs:
                accepted = accepted or r['active']
            
            if not accepted:
                count -= 1
                break
                
        tag = getAceptedTag(regexs)
        
        print(tag,text[:count])

        text = text[count:]
        