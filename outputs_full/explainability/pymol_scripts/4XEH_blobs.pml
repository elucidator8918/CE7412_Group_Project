# SoftBlobGIN blob visualization for 4XEH
# Generated automatically — open in PyMOL

fetch 4xeh, async=0
remove not chain A
hide everything
show cartoon, chain A
color gray80, chain A

# Color residues by blob assignment
select blob1, chain A and resi 30+80+137+147+152+153+166+169+173+174+196+200+207+225+237+291+292
color red, blob1
# Blob 1: 17 residues, importance=0.8398
select blob2, chain A and resi 25+28+42+53+57+107+135+165+168+172+178+187+191+193+218+219+229+241+247+262+290+294
color orange, blob2
# Blob 2: 22 residues, importance=0.0000
select blob3, chain A and resi 115+127+131+192+208+210+222+257+259+284+286+288
color yellow, blob3
# Blob 3: 12 residues, importance=0.0000
select blob4, chain A and resi 18+19+31+32+33+34+45+54+59+64+71+74+114+116+132+140+161+171+181+183+184+199+216+217+231+233+234+235+239+245+246+251+256+261+277+298+317
color green, blob4
# Blob 4: 37 residues, importance=0.0000
select blob5, chain A and resi 78+109+134+136+139+151+182+190+227+238+249+282+289+293
color cyan, blob5
# Blob 5: 14 residues, importance=0.0039
select blob6, chain A and resi 26+27+104+105+146+205+211+212+213+215+248+255+278+285+287+314
color blue, blob6
# Blob 6: 16 residues, importance=0.0117
select blob7, chain A and resi 1+2+3+4+6+7+8+9+10+11+12+13+14+15+16+17+20+21+24+35+36+37+38+39+40+43+44+47+48+49+51+52+55+58+60+61+62+63+65+66+67+68+69+70+72+73+76+82+83+84+87+88+89+90+91+92+93+94+95+96+97+98+99+100+101+103+106+108+111+112+117+118+119+120+121+122+123+124+126+128+130+142+143+144+145+150+154+155+156+157+158+159+160+163+170+176+177+189+194+195+197+198+201+202+203+206+209+214+221+223+226+228+230+236+240+242+244+250+252+253+258+264+265+267+268+269+270+271+272+273+275+276+279+281+295+296+297+299+300+301+302+303+304+305+306+307+308+309+310+311+312+315+316+318+319+320+321+322+323+324+325+326+327
color purple, blob7
# Blob 7: 163 residues, importance=0.1211
select blob8, chain A and resi 5+22+23+29+41+46+50+56+75+77+79+81+85+86+102+110+113+125+129+133+138+141+148+149+162+164+167+175+179+180+185+186+188+204+220+224+232+243+254+260+263+266+274+280+283+313
color white, blob8
# Blob 8: 46 residues, importance=0.0234

# Active/binding site residues
select active_site, chain A and resi 25+26+27+28+49+53+83+84+85+86+108+134+191+195+227+231+252
show sticks, active_site
color magenta, active_site
set stick_radius, 0.15, active_site

# Domain boundaries
# Domain 0: residues 14-178
# Domain 1: residues 184-327

# Final styling
set cartoon_transparency, 0.3
set ray_shadow, 0
bg_color white
orient
zoom chain A