from sympy.parsing.latex import parse_latex
from sympy import symbols, simplify, parse_expr, tan, sin, cos, exp

latex = r"e_{1} = \cos{\left(\tan{\left(x_{0} \sin{\left(5.67205477072444 + \frac{0.366612788118867 \left(13.4473622683011 - \frac{- \frac{2.89929305296405 So x_{8}}{\left(- x_{2} \left(- Pr + \frac{Pr}{- x_{7} \tan{\left(x_{0} \sin{\left(x_{11} \left(\frac{x_{9}}{x_{12} \sin{\left(So \right)} + x_{12} - 13.3172842630643} - 2.6796023195273\right) + 5.67205477072444 \right)} \right)} - 0.681408703494794}\right) + 13.4473622683011 - \frac{1}{- x_{5} \sin{\left(x_{6} \right)} - 0.681408703494794}\right) \sin{\left(\sin{\left(So + x_{10} \right)} \right)}} + 1.15210004181929 x_{1} - \frac{1}{- \sin{\left(x_{6} + \frac{2.7599397743342}{x_{6}} \right)} \tan{\left(x_{0} \sin{\left(5.67205477072444 - \frac{0.0659625294474778}{\tan{\left(Pr + 0.906041733350064 \right)}} \right)} \right)} - 0.681408703494794} - \frac{1}{- x_{7} \tan{\left(x_{0} \sin{\left(x_{11} \left(\frac{x_{9}}{- x_{1} \left(0.831594132340229 \cos{\left(\tan{\left(x_{0} \sin{\left(\frac{2.32048195430829}{x_{4}} \right)} \right)} \right)} + 22.5007279806321\right) \sin{\left(0.376431160840897 Pr \right)} - 21.9678342990584} - 2.6796023195273\right) + 5.67205477072444 \right)} \right)} - 0.681408703494794}}{\cos{\left(x_{3} \right)} - 16.665243113868} - \frac{1}{- x_{5} \sin{\left(0.230499839607236 x_{1} + x_{6} \right)} - 0.681408703494794}\right)}{x_{8}} \right)} \right)} \right)} + \cos{\left(\tan{\left(x_{0} \sin{\left(5.67205477072444 + \frac{0.366612788118867 \left(- x_{1} \left(Te + 0.681408703494794\right) - \frac{x_{2}}{- x_{5} \sin{\left(0.731048195397045 x_{1} \cos{\left(\cos{\left(Pr \right)} \right)} + x_{7} \right)} - 0.681408703494794} + e^{\cos{\left(Te + 1.36281740698959 \right)}}\right)}{- 2.6796023195273 x_{0} \sin{\left(\sin{\left(Te - x_{1} \left(Te - 0.932673661417481\right) \right)} \right)} - 8.98781653545383} \right)} \right)} \right)} + 16.2494069157883"

# Define symbols
So, Pr, Te = symbols('So Pr Te')

expr = parse_latex(latex)
x0 = parse_expr("1/So")
x1 = parse_expr("1/Pr")
x2 = parse_expr("1/Te")
x3 = parse_expr("Pr - 0.179924245921533")
x4 = parse_expr("tan(x3)")
x5 = tan(x0*sin(0.366612788118867*cos(x4) + 5.93473124283401))
x6 = cos(Te - 2.32048195430829)
x7 = sin(2.7599397743342*x1 + x6)
x8 = -Pr + tan(Pr + 0.98197187199823) - 2.89929305296405
x9 = 1.56052541265706*Pr
x10 = -Te
x11 = 0.366612788118867/(-Pr + tan(Pr + x10) - 0.48926022798077)
x12 = 2.6796023195273*x1

# Print the parsed expression and variables
print(f"Parsed expression: {simplify(expr)}")