from cmp.utils import Token
from my_tools.tokenizer import *

class Lexer:
    def __init__(self, table, eof):
        self.eof = eof
        self.regexs = self._build_regexs(table)
    
    def _build_regexs(self, table):
        automatonMaker = RegexHandler()
        regexs = []
        for _, (tag, regex) in enumerate(table):
            autom = automatonMaker(regex)
            regexs.append({
                "tag": tag,
                "autom":autom,
                "active":True,
                "state":[0],
                "count":0
            })
        return regexs
    
    def _reset_automs(self):
        
        for regex in self.regexs:
            regex['active'] = True
            regex['count'] = 0
            regex['state'] = {0}
    
    def _getAceptedTag(self, regexs):
        max = 0
        accepted_regex = None

        for r in regexs:
            if r['count'] > max and r['state'].issubset(r['autom'].finals):
                
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
        
    def _walk(self, text):
        
        head = 0
        
        for c in text:
            if c == ' ':
                break
            
            head += 1 

            for regex in self.regexs:
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
            for r in self.regexs:
                accepted = accepted or r['active']
            
            if not accepted:
                head -= 1
                break
                
        tag = self._getAceptedTag(self.regexs)
        
        # print(tag,text[:head])

        lex = text[:head]
        
        return lex,tag
    
    def _tokenize(self, text):
        
        while text:
            text = text.lstrip()
            self._reset_automs()
            
            lex,tag = self._walk(text)
            yield lex,tag
            text = text.lstrip(lex)
        yield '$', self.eof
    
    def __call__(self, text):
        return [ Token(lex, ttype) for lex, ttype in self._tokenize(text) ]
    
    
    
    
    
    


if __name__ == "__main__":

    nonzero_digits = '|'.join(str(n) for n in range(1,10))
    letters = '|'.join(chr(n) for n in range(ord('a'),ord('z')+1))

    # print('Non-zero digits:', nonzero_digits)
    # print('Letters:', letters)

    lexer = Lexer([
        ('num', f'({nonzero_digits})(0|{nonzero_digits})*'),
        ('for' , 'for'),
        ('foreach' , 'foreach'),
        ('id', f'({letters})({letters}|0|{nonzero_digits})*')
    ], 'eof')

    text = '5465 for 45foreach fore'
    print(f'\n>>> Tokenizando: "{text}"')
    tokens = lexer(text)
    print(tokens)

