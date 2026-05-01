# SoftBlobGIN blob visualization for 4ETL
# Generated automatically — open in PyMOL

fetch 4etl, async=0
remove not chain A
hide everything
show cartoon, chain A
color gray80, chain A

# Color residues by blob assignment
select blob1, chain A and resi 45+54+136+138+179+199+239+241+243
color red, blob1
# Blob 1: 9 residues, importance=0.7891
select blob2, chain A and resi 93+113+118+132+147+148+154+167+172+204+205+207+216+224+240+251
color orange, blob2
# Blob 2: 16 residues, importance=0.0000
select blob3, chain A and resi 139+171+180+191+203+215+219+236
color yellow, blob3
# Blob 3: 8 residues, importance=0.0000
select blob4, chain A and resi 49+60+78+86+94+111+115+143+144+162+186+197+200+221+225+229+245
color green, blob4
# Blob 4: 17 residues, importance=0.0039
select blob5, chain A and resi 52+83+103+134+140+155+176+226+235+248
color cyan, blob5
# Blob 5: 10 residues, importance=0.0117
select blob6, chain A and resi 130+159+178+190+195+228+237
color blue, blob6
# Blob 6: 7 residues, importance=0.0117
select blob7, chain A and resi 1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+21+22+23+24+25+26+27+28+29+30+31+32+33+34+35+36+37+38+39+40+41+42+43+44+46+47+48+55+56+59+62+63+64+65+66+67+68+69+70+71+72+73+74+75+76+79+81+82+84+85+87+88+89+90+92+95+96+97+98+99+100+101+102+104+105+107+108+110+117+119+120+121+122+123+125+127+128+129+131+133+135+137+146+150+151+153+157+158+160+161+163+164+166+169+170+173+174+175+177+181+182+183+184+185+187+188+189+192+193+194+196+208+209+211+212+213+214+217+220+223+230+231+232+233+234+238+246+247+249+250+252+253+254+255+256+257+258+259+260+261+262+263+264+265+266+267+268+269+271+272+273+274+275+276+277+278+279+280
color purple, blob7
# Blob 7: 178 residues, importance=0.1602
select blob8, chain A and resi 50+51+53+57+58+61+77+80+91+106+109+112+114+116+124+126+141+142+145+149+152+156+165+168+198+201+202+206+210+218+222+227+242+244+270
color white, blob8
# Blob 8: 35 residues, importance=0.0234

# Active/binding site residues
select active_site, chain A and resi 143+148+189
show sticks, active_site
color magenta, active_site
set stick_radius, 0.15, active_site

# Domain boundaries
# Domain 0: residues 16-264

# Final styling
set cartoon_transparency, 0.3
set ray_shadow, 0
bg_color white
orient
zoom chain A