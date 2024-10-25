from logic import *

rain = Symbol("rain")
bbc = Symbol("bbc")
unimayor = Symbol("unimayor")

knowledge = And(
    Implication(Not(rain), bbc),  
    Or(bbc, unimayor),            
    Not(And(bbc, unimayor)),      
    unimayor                       
)

print("¿Los estudiantes visitaron BBC?", model_check(knowledge, bbc))
print("¿Está lloviendo?", model_check(knowledge, rain))