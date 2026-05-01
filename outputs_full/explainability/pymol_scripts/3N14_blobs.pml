# SoftBlobGIN blob visualization for 3N14
# Generated automatically — open in PyMOL

fetch 3n14, async=0
remove not chain A
hide everything
show cartoon, chain A
color gray80, chain A

# Color residues by blob assignment
select blob1, chain A and resi 17+98+111+131+175+178+181+185+186+229+232+326
color red, blob1
# Blob 1: 12 residues, importance=0.7500
select blob2, chain A and resi 21+48+55+64+68+125+136+147+174+200+209+227+261+265+271+294+299+321+335
color orange, blob2
# Blob 2: 19 residues, importance=0.0000
select blob3, chain A and resi 63+100+104+108+110+264+327
color yellow, blob3
# Blob 3: 7 residues, importance=0.0000
select blob4, chain A and resi 22+23+25+26+27+32+47+61+65+93+105+113+135+140+142+156+159+166+180+187+188+191+231+242+249+263+270+281+286+287+309+324+332
color green, blob4
# Blob 4: 33 residues, importance=0.0156
select blob5, chain A and resi 57+115+130+143+146+179+323
color cyan, blob5
# Blob 5: 7 residues, importance=0.0000
select blob6, chain A and resi 56+67+70+127+128+129+182+190+206+262+268+331
color blue, blob6
# Blob 6: 12 residues, importance=0.0156
select blob7, chain A and resi 1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+18+19+24+28+30+31+33+34+35+36+37+38+39+40+41+42+43+44+45+49+50+51+52+54+58+59+60+62+66+69+71+74+75+76+78+79+80+81+82+83+84+86+87+89+90+91+92+94+95+96+101+103+106+107+118+119+120+121+122+123+124+126+134+137+138+139+141+144+148+149+150+151+152+153+154+155+157+158+160+161+163+165+167+168+169+171+172+173+176+183+192+193+194+196+197+199+202+203+205+210+211+212+213+214+215+218+219+220+221+222+223+224+225+226+228+233+235+236+237+238+239+240+241+243+244+245+246+247+248+253+254+255+256+257+258+259+267+272+273+274+275+276+277+278+279+280+282+284+285+288+289+290+291+292+293+296+297+300+302+303+304+305+306+308+310+311+312+313+314+315+316+317+318+319+320+322+325+328+329+330+334+336+337+338+339+340+342+343+344+345+346+347+348+349+351+352+353+354+355+356+357+358+359
color purple, blob7
# Blob 7: 219 residues, importance=0.1797
select blob8, chain A and resi 20+29+46+53+72+73+77+85+88+97+99+102+109+112+114+116+117+132+133+145+162+164+170+177+184+189+195+198+201+204+207+208+216+217+230+234+250+251+252+260+266+269+283+295+298+301+307+333+341+350
color white, blob8
# Blob 8: 50 residues, importance=0.0391

# Active/binding site residues
select active_site, chain A and resi 23+25+57+99+178+181+231+302+303+325+326
show sticks, active_site
color magenta, active_site
set stick_radius, 0.15, active_site

# Domain boundaries
# Domain 0: residues 2-359

# Final styling
set cartoon_transparency, 0.3
set ray_shadow, 0
bg_color white
orient
zoom chain A