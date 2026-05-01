# SoftBlobGIN blob visualization for 1E7R
# Generated automatically — open in PyMOL

fetch 1e7r, async=0
remove not chain A
hide everything
show cartoon, chain A
color gray80, chain A

# Color residues by blob assignment
select blob1, chain A and resi 9+17+58+62+63+86+100+131+138+140+159+242+243
color red, blob1
# Blob 1: 13 residues, importance=-0.0044
select blob2, chain A and resi 14+39+82+102+104+106+107+122+142+163
color orange, blob2
# Blob 2: 10 residues, importance=-0.0068
select blob3, chain A and resi 7+11+73+83+141+210+239
color yellow, blob3
# Blob 3: 7 residues, importance=-0.0030
select blob4, chain A and resi 8+16+40+61+79+84+85+145+147+181+198+206+214+220+271+282+295
color green, blob4
# Blob 4: 17 residues, importance=-0.0059
select blob5, chain A and resi 57+76+77+89+101+127+133+137+240+283
color cyan, blob5
# Blob 5: 10 residues, importance=0.0015
select blob6, chain A and resi 75+87+132+135+146+158+161
color blue, blob6
# Blob 6: 7 residues, importance=-0.0016
select blob7, chain A and resi 1+2+3+4+5+6+10+18+19+20+21+22+23+24+25+26+27+28+29+30+31+32+33+34+35+36+37+38+41+42+43+44+45+46+47+48+49+50+51+52+53+54+55+56+59+60+64+65+66+69+71+72+74+78+88+90+91+92+93+94+95+96+97+98+108+110+111+112+113+114+116+117+118+119+120+121+123+124+125+126+129+134+139+143+150+151+152+153+164+165+167+168+169+170+171+172+173+174+175+176+177+178+179+180+182+183+184+185+186+187+188+190+191+192+193+194+196+197+199+200+201+203+204+205+207+211+212+213+215+217+219+221+222+223+224+225+226+227+228+229+230+231+232+233+234+235+236+237+238+241+245+247+249+250+251+252+253+254+256+257+258+260+261+263+264+265+267+268+269+272+273+274+275+277+278+279+280+281+284+285+286+287+288+289+290+292+293+294+296+297+298+299+300+301+303+304+305+306+308+309+311+312+313+314
color purple, blob7
# Blob 7: 204 residues, importance=0.0718
select blob8, chain A and resi 12+13+15+67+68+70+80+81+99+103+105+109+115+128+130+136+144+148+149+154+155+156+157+160+162+166+189+195+202+208+209+216+218+244+246+248+255+259+262+266+270+276+291+302+307+310
color white, blob8
# Blob 8: 46 residues, importance=-0.0014

# Domain boundaries
# Domain 0: residues 6-246

# Final styling
set cartoon_transparency, 0.3
set ray_shadow, 0
bg_color white
orient
zoom chain A