# SoftBlobGIN blob visualization for 2VDG
# Generated automatically — open in PyMOL

fetch 2vdg, async=0
remove not chain A
hide everything
show cartoon, chain A
color gray80, chain A

# Color residues by blob assignment
select blob1, chain A and resi 13+15+41+42+45+108+176+240+241+244+254
color red, blob1
# Blob 1: 11 residues, importance=0.8477
select blob2, chain A and resi 20+52+77+127+151+154+183+187+228+243+249+251+252+255+258+276+278+289
color orange, blob2
# Blob 2: 18 residues, importance=0.0000
select blob3, chain A and resi 44+74+107+111+260
color yellow, blob3
# Blob 3: 5 residues, importance=0.0000
select blob4, chain A and resi 6+30+58+76+136+155+161+177+191+203+212+214+219+220+223+257
color green, blob4
# Blob 4: 16 residues, importance=0.0039
select blob5, chain A and resi 4+50+109+157+172+188+196+232+239+259+270+281
color cyan, blob5
# Blob 5: 12 residues, importance=0.0156
select blob6, chain A and resi 17+22+197+198+210+213+231+233+247+256+275+277+279
color blue, blob6
# Blob 6: 13 residues, importance=0.0117
select blob7, chain A and resi 1+3+5+7+8+9+10+11+12+14+16+18+21+23+24+25+26+27+28+31+33+34+35+36+37+38+40+46+47+48+49+51+55+57+59+60+61+62+63+64+65+66+67+68+69+70+71+73+78+80+81+83+84+85+86+87+88+89+90+91+92+93+94+95+96+97+98+99+100+101+102+104+105+110+113+114+115+116+117+118+119+120+121+122+123+124+125+126+128+129+131+132+133+134+135+137+138+139+140+141+142+143+144+145+146+147+149+150+153+156+159+162+163+164+165+166+167+168+169+170+173+174+178+179+180+181+182+184+185+192+193+194+195+199+200+201+202+204+205+206+207+211+215+216+217+218+221+224+225+226+227+229+230+234+237+238+242+245+246+248+250+261+262+265+266+267+269+272+273+285+286+287+288+291+292+293+294+295+296+297+298+299+300+301+302+303+305+306+307+308
color purple, blob7
# Blob 7: 190 residues, importance=0.0938
select blob8, chain A and resi 2+19+29+32+39+43+53+54+56+72+75+79+82+103+106+112+130+148+152+158+160+171+175+186+189+190+208+209+222+235+236+253+263+264+268+271+274+280+282+283+284+290+304
color white, blob8
# Blob 8: 43 residues, importance=0.0273

# Active/binding site residues
select active_site, chain A and resi 84+112+145+239+240+241+242+243+244+245+246+247+248+249+250+251+252+253+254+255+256+257+258+259+260+261+262+263+264+265+266+267+268+269+270+271+272+273+274+275+276+277+278+279+280+281+282+283+284+285+286+287+288+289+290+291+292+293
show sticks, active_site
color magenta, active_site
set stick_radius, 0.15, active_site

# Domain boundaries
# Domain 0: residues 37-308

# Final styling
set cartoon_transparency, 0.3
set ray_shadow, 0
bg_color white
orient
zoom chain A