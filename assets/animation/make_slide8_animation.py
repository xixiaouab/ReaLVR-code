"""A 54-second, deterministic film built from ReaLVR's original slide 8.

Default: regenerate ../evidence-credit.svg.
Static frame: python make_slide8_animation.py --frame 12.5 --output /tmp/frame.svg
The CSS animation and static exporter evaluate the same piecewise-linear tracks.
No browser, raster renderer, or animation clock is needed for frame export.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import base64
import html
import os
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen

ASSETS = HERE / 'inputs'
SOURCE = ASSETS / 'slide-8-cropped.pdf'
NATIVE = HERE / '.cache' / 'slide-8-native.svg'
COMIC_FONT = Path(os.environ.get('COMIC_FONT', '/System/Library/Fonts/Supplemental/Comic Sans MS.ttf'))
COMIC_BOLD_FONT = Path(os.environ.get('COMIC_BOLD_FONT', '/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf'))
W, H, SECONDS = 1200, 724, 42
DURATION = 54
# The extra screen time is added after each action has settled, for reading.
TIME_MAP = [(0,0),(5.7,5.7),(9,10),(17,18),(18.2,22),(22.8,26.6),(25.3,30.5),(28.6,33.8),(31.1,38),(36.5,43.4),(38.2,49),(42,54)]
def warp_time(t):
    return interpolate([(a,(b,)) for a,b in TIME_MAP],t)[0]
def source_time(t):
    return interpolate([(b,(a,)) for a,b in TIME_MAP],t)[0]
ORANGE, BLUE, PURPLE = '#ff9900', '#6395ff', '#c46bc7'
INK, GRAY = '#252525', '#666666'
CHAPTERS = [
    {'time': 0, 'label': 'Image & question → tokens'},
    {'time': 10, 'label': 'Free-running latent reasoning'},
    {'time': 22, 'label': 'Read the same latent span'},
    {'time': 30.5, 'label': 'Assign differential credit'},
    {'time': 38, 'label': 'Ground in visual evidence'},
    {'time': 49, 'label': 'The original main figure'},
]

def esc(s): return html.escape(str(s), quote=True)

# Store visible lettering as outlines: local font availability must not change
# the diagram or its captions in a visitor's browser. No font file is served.
COMIC_GLYPHS = {}
SCRIPT_GLYPHS = {'⁺': ('+', .60, .38), '⁻': ('−', .60, .38),
                 '₁': ('1', .60, -.18), '₅': ('5', .60, -.18),
                 '₊': ('+', .60, -.18)}

@lru_cache(maxsize=2)
def comic_font(weight):
    font = TTFont(COMIC_BOLD_FONT if weight == 'bold' else COMIC_FONT)
    kern = {}
    if 'kern' in font:
        for table in font['kern'].kernTables:
            if table.version == 0 and table.coverage & 1:
                kern.update(table.kernTable)
    return font, font.getGlyphSet(), font.getBestCmap(), kern

@lru_cache(maxsize=256)
def comic_glyph(weight, char):
    font, glyphs, cmap, _ = comic_font(weight)
    if ord(char) not in cmap:
        raise ValueError(f'Comic Sans MS lacks {char!r} (U+{ord(char):04X})')
    name = cmap[ord(char)]
    pen = SVGPathPen(glyphs)
    glyphs[name].draw(pen)
    ident = f'comic-{weight}-{ord(char):x}'
    COMIC_GLYPHS[ident] = f'<path id="{ident}" d="{pen.getCommands()}"/>'
    return ident, name, font['hmtx'][name][0]

def text(x, y, value, size=28, color=INK, anchor='start', weight='normal', **attrs):
    value = ' '.join(str(value).split())
    font, _, _, kern = comic_font(weight)
    upem = font['head'].unitsPerEm
    scale = size / upem
    cursor, previous, letters = 0, None, []
    for char in value:
        base, factor, shift = SCRIPT_GLYPHS.get(char, (char, 1, 0))
        ident, name, advance = comic_glyph(weight, base)
        if previous is not None and factor == 1:
            cursor += kern.get((previous, name), 0)
        letters.append(f'<use href="#{ident}" xlink:href="#{ident}" transform="translate({cursor:g} {shift*upem:g}) scale({factor:g})"/>')
        cursor += advance * factor
        previous = name if factor == 1 else None
    offset = {'start': 0, 'middle': cursor/2, 'end': cursor}[anchor] * scale
    extra=' '.join(f'{k.replace("_", "-")}="{esc(v)}"' for k,v in attrs.items())
    return f'<g class="comic-text" role="img" aria-label="{esc(value)}" data-font="Comic Sans MS" fill="{color}" transform="translate({x-offset:g} {y:g}) scale({scale:g} {-scale:g})" {extra}>{"".join(letters)}</g>'
def rect(x,y,w,h,fill='#fff',stroke='none',sw=2,rx=7,**attrs):
    extra=' '.join(f'{k.replace("_", "-")}="{esc(v)}"' for k,v in attrs.items())
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'
def path(d,stroke=INK,width=2.5,**attrs):
    extra=' '.join(f'{k.replace("_", "-")}="{esc(v)}"' for k,v in attrs.items())
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" {extra}/>'
def image_ref(name,x,y,w,h):
    return f'<use href="#{name}" xlink:href="#{name}" transform="translate({x} {y}) scale({w} {h})"/>'
def original_crop(x,y,w,h,box):
    return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="{" ".join(map(str,box))}" overflow="hidden"><use href="#original-art" xlink:href="#original-art"/></svg>'
def chip(label='',kind='latent',w=65,h=65,outline=False):
    fill={'image':ORANGE,'text':BLUE,'latent':'url(#latent-fill)','special':'white'}[kind]
    s=rect(2,4,w,h,'#000',rx=6,opacity='.08')+rect(0,0,w,h,fill,INK if kind=='special' else 'none',2.5,6)
    if outline: s+=rect(-6,-6,w+12,h+12,'none',ORANGE,3,11)
    if label: s+=text(w/2,h*.67,label,30 if kind=='special' else 23,INK if kind!='text' else '#fff','middle')
    return s

@lru_cache(maxsize=1)
def original_svg():
    if not NATIVE.exists() or SOURCE.stat().st_mtime > NATIVE.stat().st_mtime:
        NATIVE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['pdftocairo', '-svg', str(SOURCE), str(NATIVE)], check=True)
    body=re.sub(r'^.*?<svg[^>]*>','',NATIVE.read_text(),count=1,flags=re.S)
    return re.sub(r'</svg>\s*$','',body)

@lru_cache(maxsize=8)
def asset_data(name):
    return base64.b64encode((ASSETS/name).read_bytes()).decode()

def interpolate(points,t):
    points=sorted(points)
    if t<=points[0][0]:return points[0][1]
    for (a,va),(b,vb) in zip(points,points[1:]):
        if a<=t<=b:
            f=(t-a)/(b-a) if b>a else 1
            return tuple(x+(y-x)*f for x,y in zip(va,vb))
    return points[-1][1]

def smooth_points(points):
    """Smooth movement, but retain identical opacity timing and static samples."""
    result=[]
    for (a,va),(b,vb) in zip(points,points[1:]):
        result.append((a,va))
        if va[:3]!=vb[:3]:
            for f in (.125,.25,.375,.5,.625,.75,.875):
                q=f*f*(3-2*f)
                state=tuple(x+(y-x)*(q if i<3 else f) for i,(x,y) in enumerate(zip(va,vb)))
                result.append((a+(b-a)*f,state))
    result.append(points[-1]);return result

class Film:
    def __init__(self,frame=None):self.frame=None if frame is None else source_time(frame);self.css=[];self.n=0
    def motion(self,content,points,name=None,smooth=True):
        self.n+=1; name=name or f'motion-{self.n}'
        points=sorted(points)
        if points[0][0]>0:points.insert(0,(0,points[0][1]))
        if points[-1][0]<SECONDS:points.append((SECONDS,points[-1][1]))
        if smooth:points=smooth_points(points)
        if self.frame is not None:
            x,y,s,o=interpolate(points,self.frame)
            return f'<g transform="translate({x:.5f} {y:.5f}) scale({s:.5f})" opacity="{o:.5f}">{content}</g>'
        frames=''.join(f'{100*warp_time(t)/DURATION:.6f}%{{transform:translate({x:.5f}px,{y:.5f}px) scale({s:.5f});opacity:{o:.5f}}}' for t,(x,y,s,o) in points)
        self.css.append(f'.{name}{{transform-origin:0 0;animation:{name} {DURATION}s linear infinite}}@keyframes {name}{{{frames}}}')
        return f'<g class="{name}">{content}</g>'
    def visible(self,content,a,b,fade=.28,name=None):
        points=[(0,(0,0,1,0))]
        if a==0:points=[(0,(0,0,1,1))]
        else:points.extend([(a,(0,0,1,0)),(a+fade,(0,0,1,1))])
        if b == SECONDS:
            # Hold the complete figure through the final export frame.
            points.append((SECONDS,(0,0,1,1)))
        else:
            points.extend([(b-fade,(0,0,1,1)),(b,(0,0,1,0)),(SECONDS,(0,0,1,0))])
        return self.motion(content,points,name,smooth=False)
    def appear(self,content,a,x=0,y=0,s=1,end=SECONDS):
        return self.motion(content,[(0,(x,y+10,s,0)),(a,(x,y+10,s,0)),(a+.28,(x,y,s,1)),(end,(x,y,s,1))])
    def flight(self,content,a,b,start,end,hold=None,fade=None):
        x,y,s=start;xx,yy,ss=end
        points=[(0,(x,y,s,0)),(a,(x,y,s,0)),(a+.15,(x,y,s,1)),(b,(xx,yy,ss,1))]
        if hold:points.append((hold,(xx,yy,ss,1)))
        if fade:points.append((fade,(xx,yy,ss,0)))
        return self.motion(content,points)
    def pulse(self,content,a,b,x=0,y=0,s=1):
        return self.motion(content,[(0,(x,y,s,0)),(a,(x,y,s,0)),(a+.22,(x,y,s,1)),(b-.22,(x,y,s,1)),(b,(x,y,s,0))])
    def caption(self,title,subtitle,a,b):
        body=rect(36,638,1128,80,'#202328',rx=9,opacity='.80')
        body+=text(600,670,title,29,'#fff','middle')+text(600,701,subtitle,22,'#fff','middle')
        return self.visible(body,a,b,.22)
    def heading(self,n,title,training=False):
        s=text(50,53,f'{n:02d}',22,'#999')+text(103,55,title,32,INK,weight='bold')+path('M50 77H1150','#ddd',1)
        if training:s+=rect(1008,25,143,36,'#fff7e7','#dfb75a',1,7)+text(1079,50,'Training only',19,'#86630f','middle')
        return s

    def scene_inputs(self):
        a=0;b=9.4
        s=self.heading(1,'See the image. Read the question.')
        s+=self.visible(image_ref('scene-photo',100,161,360,242),0,5.15,.3)
        s+=self.visible(rect(98,159,364,246,'none','#aeb7bb',1.5,3),0,5.15,.3)
        # Original robot silently inspects the source scene, with a gentle nod.
        s+=self.motion(image_ref('robot',0,0,130,130),[(0,(33,393,1,1)),(2,(40,385,1,1)),(4,(34,393,1,1)),(6,(38,389,1,1)),(9.4,(38,389,1,1))])
        s+=self.visible(text(280,138,'Image',27,INK,'middle')+text(845,138,'Question',27,INK,'middle'),0,5.2,.25)
        chunks=['What are','the last four','digits','of the','phone number','on the banner?']
        for i,word in enumerate(chunks):
            yy=160+i*46
            xx=662
            tokenx=650+i*60
            # A word fragment moves into its blue embedding position.
            group=text(0,30,word,29)
            s+=self.motion(group,[(0,(xx,yy,1,1)),(2.2+i*.35,(xx,yy,1,1)),(2.6+i*.35,(xx,yy,.23,.9)),(3.2+i*.35,(tokenx,484,.23,.8)),(3.45+i*.35,(tokenx,484,.23,0))])
            s+=self.motion(chip('', 'text',42,42),[(0,(tokenx,485,1,0)),(3.1+i*.35,(tokenx,485,1,0)),(3.45+i*.35,(tokenx,485,1,1)),(9.4,(tokenx,485,1,1))])
        # Every image patch visibly peels away, contracts, and becomes a token.
        for i in range(6):
            col,row=i%3,i//3
            crop=f'<svg width="120" height="121" viewBox="{col/3} {row/2} {1/3} .5" preserveAspectRatio="none"><use href="#scene-photo" xlink:href="#scene-photo"/></svg>'
            tile=rect(-2,-2,124,125,'white','#fff',2,2)+crop
            sx,sy=100+col*120,161+row*121;tx=175+i*60
            s+=self.motion(tile,[(0,(sx,sy,1,0)),(1.75+i*.15,(sx,sy,1,0)),(1.95+i*.15,(sx,sy,1,1)),(2.6+i*.15,(sx+(col-1)*15,sy+(row-.5)*16,1,1)),(4+i*.2,(tx,485,.35,1)),(4.3+i*.2,(tx,485,.35,0))])
            s+=self.motion(chip('', 'image',42,42),[(0,(tx,485,1,0)),(3.8+i*.2,(tx,485,1,0)),(4.2+i*.2,(tx,485,1,1)),(9.4,(tx,485,1,1))])
        # Clear persistent token names, after the images/words have transformed.
        s+=self.appear(text(346,575,'Visual tokens',29,INK,'middle'),5.4)
        s+=self.appear(text(821,575,'Text tokens',29,INK,'middle'),5.4)
        s+=self.visible(path('M175 543v9h342v-9','#888',1.8)+path('M650 543v9h342v-9','#888',1.8),5.3,b,.3)
        return self.visible(s,a,b,.28)

    def scene_rollout(self):
        a=9;b=18.6
        s=self.heading(2,'Let the model reason in latent space.')
        s+=rect(86,245,1028,91,'#ededed',INK,2.5,7)+text(600,301,'Multi-Modal Large Language Model',34,INK,'middle','bold')
        s+=text(334,132,'Image tokens',23,GRAY,'middle')+text(824,132,'Question tokens',23,GRAY,'middle')
        for kind,startx in [('image',175),('text',650)]:
            for i in range(6):
                xx=startx+i*60
                s+=self.motion(chip('',kind,42,42),[(0,(xx,161,1,1)),(9.7+i*.15,(xx,161,1,1)),(10.65+i*.15,(510+(i-2.5)*10,241,.56,1)),(10.85+i*.15,(510+(i-2.5)*10,247,.56,0))])
        s+=self.visible(text(600,214,'image + question',23,GRAY,'middle'),9.3,11.4,.2)
        s+=self.appear(chip('S','special',46,62),11.2,153,432)
        s+=self.appear(text(176,525,'start',21,GRAY,'middle'),11.2)
        xs=[258,394,530,666,802]
        for i,x in enumerate(xs):
            t=11.8+i*.65
            s+=self.flight(chip(f'z{i+1}','latent',67,67),t,t+.57,(567,337,.5),(x,429,1))
            if i<4:
                s+=self.pulse(path(f'M{x+68} 462H{xs[i+1]-13}',INK,2,marker_end='url(#arrow)'),t+.5,t+1.1)
                # A tiny rectangular recurrent state returns through the model.
                s+=self.motion(chip('','latent',20,20),[(0,(x+24,426,1,0)),(t+.52,(x+24,426,1,0)),(t+.63,(x+24,410,1,1)),(t+.91,(600,350,.7,1)),(t+1.03,(600,335,.7,0))])
        s+=self.appear(text(600,401,'Free-running latent rollout',26,INK,'middle'),11.5)
        s+=self.appear(chip('E','special',46,62),15.1,920,432)
        s+=self.appear(text(943,525,'end',21,GRAY,'middle'),15.1)
        # Only after E does ordinary answer decoding appear.
        s+=self.appear(text(444,588,'Answer',27,INK,'end'),15.7)
        for i,digit in enumerate('8015'):
            s+=self.flight(chip(digit,'text',43,43),15.75+i*.23,16.35+i*.23,(929,433,.5),(489+i*58,553,1))
        s+=self.pulse(path('M952 499V574H754',BLUE,2.4,marker_end='url(#arrow-blue)'),15.6,17.65)
        return self.visible(s,a,b,.28)

    def scene_readouts(self):
        a=18.2;b=25.7
        s=self.heading(3,'Read one shared latent span twice.',True)
        s+=text(600,129,'The same saved latent span',28,INK,'middle')
        xs=[300,427,554,681,808]
        for i,x in enumerate(xs):s+=chip_at(x,158,f'z{i+1}')
        s+=path('M305 242v10H870v-10',PURPLE,2)+text(724,279,'shared z₁ … z₅',22,PURPLE,'middle')
        s+=rect(177,312,846,62,'#ededed',INK,2,7)+text(600,353,'Decoder attention',30,INK,'middle','bold')
        # The two branches start only at the shared, already-generated span.
        s+=self.pulse(path('M600 254V303',PURPLE,3,marker_end='url(#arrow-purple)'),18.8,24.9)
        for xx,digits,color,ta in [(190,'8015','#0072bc',19.4),(720,'7915','#e33131',20.8)]:
            s+=rect(xx,428,290,94,'none',color,2,5,stroke_dasharray='8 7')
            s+=text(xx+145,413,'Correct answer' if digits=='8015' else 'Wrong answer',27,color,'middle')
            s+=self.pulse(path(f'M{xx+145} 376V417',color,2.6,marker_end='url(#arrow-blue)' if color=='#0072bc' else 'url(#arrow-red)'),ta,ta+2.25)
            for j,d in enumerate(digits):s+=self.appear(text(xx+87+j*39,489,d,43,color,'middle','bold'),ta+.45+j*.18)
        s+=self.appear(text(600,563,'Teacher-forced answer readouts',28,INK,'middle'),21.9)
        s+=self.appear(text(600,603,'r⁺ from 8015    ·    r⁻ from 7915',25,GRAY,'middle'),22.25)
        return self.visible(s,a,b,.28)

    def scene_credit(self):
        a=25.3;b=31.5
        s=self.heading(4,'Give useful latent positions more credit.',True)
        s+=text(600,134,'Compare attention readouts:  [ r⁺ − r⁻ ]₊',30,INK,'middle')
        xs=[294,422,550,678,806]
        plus=[.35,.83,.96,.36,.42];minus=[.35,.32,.27,.36,.42]
        s+=text(205,220,'r⁺',31,'#0072bc','middle')+text(205,300,'r⁻',31,'#e33131','middle')
        for i,x in enumerate(xs):
            s+=rect(x,179,74,45,'#edf3fb',rx=3)+rect(x,259,74,45,'#faeded',rx=3)
            s+=self.motion(rect(0,0,74,45*plus[i],'#79aada',rx=2),[(0,(x,224,1,0)),(25.7,(x,224,1,0)),(26.2,(x,224-45*plus[i],1,1))])
            s+=self.motion(rect(0,0,74,45*minus[i],'#e79a9a',rx=2),[(0,(x,304,1,0)),(26.15,(x,304,1,0)),(26.65,(x,304-45*minus[i],1,1))])
            high=i in (1,2)
            s+=self.flight(rect(0,0,16,high and 24 or 8,ORANGE,rx=2),26.8+i*.12,27.5+i*.12,(x+27,326,1),(x+27,407,1),hold=28.2,fade=28.6)
            s+=chip_at(x+3,466,f'z{i+1}')
            # Visible baseline for ALL positions, stronger detached weight for two.
            s+=self.appear(rect(x,445,74,8,'#e8b569',rx=2),27.6)
            if high:
                h=42 if i==1 else 52
                s+=self.motion(rect(0,0,74,h,'#ffc565',rx=3),[(0,(x,445,1,0)),(27.5,(x,445,1,0)),(28,(x,445-h,1,1))])
                s+=self.pulse(rect(x-5,458,85,84,'none',ORANGE,3,12),28,31.4)
        s+=self.appear(text(600,363,'Detached token weights',27,PURPLE,'middle'),27.4)
        s+=self.appear(text(600,587,'More credit here; a baseline at every position.',27,INK,'middle'),28.3)
        s+=text(1110,613,'Schematic weights',17,'#888','end')
        return self.visible(s,a,b,.28)

    def scene_grounding(self):
        a=31.1;b=38.6
        s=self.heading(5,'Ground the reasoning in visual evidence.',True)
        s+=text(365,131,'Relevant ROI  p⁺',27,'#0072bc','middle')+text(838,131,'Mismatched ROI  p⁻',27,'#e33131','middle')
        # Both crops come from the unchanged main illustration.
        s+=rect(215,151,300,109,'none','#0072bc',2,6,stroke_dasharray='8 6')
        s+=rect(688,151,300,109,'none','#e33131',2,6,stroke_dasharray='8 6')
        s+=image_ref('positive-photo',228,163,274,83)
        s+=original_crop(701,163,274,83,(468,87.5,169,49))
        s+=self.motion(image_ref('robot',0,0,119,119),[(0,(46,393,1,1)),(32,(50,389,1,1)),(34,(46,393,1,1)),(36,(50,389,1,1)),(38.6,(46,393,1,1))])
        xs=[323,438,553,668,783]
        for i,x in enumerate(xs):
            s+=chip_at(x,334,f'z{i+1}')
            if i in (1,2):s+=rect(x-6,328,79,79,'none',ORANGE,3,11)
        s+=text(600,305,'The same latent positions, with their weights',24,GRAY,'middle')
        s+=self.pulse(path('M365 262C365 284 472 274 472 320','#0072bc',2.5,marker_end='url(#arrow-blue)'),31.85,34.8)
        s+=self.pulse(path('M590 322C590 278 838 290 838 266','#e33131',2.5,marker_end='url(#arrow-red)',stroke_dasharray='7 6'),32.4,35.3)
        # Small crop cards travel with the corresponding visual evidence.
        s+=self.flight(image_ref('positive-photo',0,0,137,42),32,33.8,(278,193,1),(435,417,.40),hold=34.1,fade=34.4)
        s+=self.flight(original_crop(0,0,137,42,(468,87.5,169,49)),32.6,34.4,(770,193,1),(655,417,.40),hold=34.7,fade=35)
        s+=rect(293,447,614,76,'#fffaf0','#d4ae4f',1.8,6,stroke_dasharray='7 5')
        s+=text(600,480,'Weighted visual contrastive loss',29,INK,'middle','bold')
        s+=text(600,510,'align with p⁺  ·  separate from p⁻',23,GRAY,'middle')
        for i,x in enumerate(xs):
            s+=self.flight(rect(0,0,12,18,'#efb466',rx=2),34+i*.11,34.6+i*.11,(x+27,406,1),(570+i*12,444,.65),hold=34.8+i*.11,fade=35+i*.11)
        s+=rect(265,572,670,46,'#ededed',INK,2,5)+text(600,603,'Multi-Modal Large Language Model',27,INK,'middle','bold')
        s+=self.pulse(path('M600 525V563',ORANGE,3.5,marker_end='url(#arrow-orange)'),35.4,38.1)
        s+=self.visible(text(819,554,'through latent generation',20,GRAY,'middle'),35.5,38.3,.25)
        s+=self.motion(rect(0,0,15,21,ORANGE,rx=3),[(0,(592,521,1,0)),(35.6,(592,521,1,0)),(35.8,(592,522,1,1)),(36.5,(592,564,1,1)),(36.7,(592,566,1,0))])
        s+=self.pulse(rect(259,566,682,58,'none',ORANGE,2.7,9),36.3,38.35)
        return self.visible(s,a,b,.28)

    def render(self):
        artwork=original_svg()
        defs=[f'<g id="original-art">{artwork}</g>','<linearGradient id="latent-fill" x1="0" y1="0" x2="0.8" y2="1"><stop stop-color="#fff4ee"/><stop offset="1" stop-color="#f0aa87"/></linearGradient>']
        for name,file in [('scene-photo','evidence-scene.png'),('positive-photo','evidence-roi.png'),('robot','realvr-robot-original.png')]:
            data=asset_data(file)
            defs.append(f'<image id="{name}" width="1" height="1" preserveAspectRatio="none" xlink:href="data:image/png;base64,{data}"/>')
        for ident,color in [('arrow',INK),('arrow-blue','#0072bc'),('arrow-red','#e33131'),('arrow-purple',PURPLE),('arrow-orange',ORANGE)]:
            defs.append(f'<marker id="{ident}" markerWidth="8" markerHeight="8" refX="6.5" refY="3.5" orient="auto" markerUnits="strokeWidth"><path d="M0 .4L6.5 3.5L0 6.6" fill="none" stroke="{color}" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></marker>')
        scene=self.scene_inputs()+self.scene_rollout()+self.scene_readouts()+self.scene_credit()+self.scene_grounding()
        overview=original_crop(0,0,1200,636,(0,0,945,501))
        scene+=self.visible(overview,38.2,42,.4)
        caps=[
            ('Turn images and words into tokens','These tokens become the model’s input.',0,9.4),
            ('Let the model think before it answers','Latent tokens form a chain, then the answer appears.',9,18.6),
            ('Read the same reasoning in two ways','Compare how the two answers read the same tokens.',18.2,25.7),
            ('Give the most useful tokens more credit','Two positions stand out. Every token still receives supervision.',25.3,31.5),
            ('Learn to preserve visual evidence','Stronger guidance for selected tokens; every position still learns.',31.1,38.6),
            ('All the pieces come together','Visual grounding and token credit work together during training.',38.2,42),
        ]
        captions=''.join(self.caption(*c) for c in caps)
        static=overview+rect(0,636,1200,88,'#fffaf3',rx=0)+text(600,674,'The ReaLVR training pipeline',31,INK,'middle')+text(600,709,'Original main figure · visual evidence and differential readout credit',22,GRAY,'middle')
        defs.extend(COMIC_GLYPHS.values())
        style='svg{overflow:hidden}.paused *{animation-play-state:paused!important}.reduced-poster{display:none}'
        style+=''.join(self.css)
        style+='@media(prefers-reduced-motion:reduce){:root:not(.motion-enabled) .film{display:none}:root:not(.motion-enabled) .reduced-poster{display:inline}:root:not(.motion-enabled) *{animation:none!important}}'
        desc='A 54-second film using the original ReaLVR slide artwork. Image patches and question words turn into rectangular tokens and enter the model. A free-running latent span precedes the decoded answer. During training, correct and wrong teacher-forced readouts share the same saved span; detached differential credit weights every position with a baseline and emphasizes two schematic positions. Relevant and mismatched visual evidence form a weighted objective that trains the recurrent latent computation. The unchanged original main figure closes the film. No training branches are added at inference.'
        return f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1200" height="724" viewBox="0 0 1200 724" role="img" aria-labelledby="title desc" data-duration="54"><title id="title">ReaLVR — from visual input to grounded latent reasoning</title><desc id="desc">{esc(desc)}</desc><metadata id="realvr-timeline">{esc(json.dumps(CHAPTERS,ensure_ascii=False))}</metadata><defs>{"".join(defs)}</defs><style>{style}</style><rect width="1200" height="724" fill="white"/><g class="film">{scene}{captions}</g><g class="reduced-poster">{static}</g></svg>'

def chip_at(x,y,label):return f'<g transform="translate({x} {y})">{chip(label)}</g>'

def render_svg(t=None):
    """Return the animation, or a static SVG at playback time t (0–54s)."""
    return Film(t).render()

def make_svg(frame=None):return render_svg(frame)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame',type=float,help='Sample the exact authored timeline, in seconds.')
    parser.add_argument('--output',type=Path,help='Output SVG; default is animated website asset.')
    parser.add_argument('--frames-dir',type=Path,help='Write six representative static SVG frames here.')
    args=parser.parse_args()
    if args.frames_dir:
        args.frames_dir.mkdir(parents=True,exist_ok=True)
        for name,t in [('inputs',3.5),('tokens',8),('rollout',20),('readouts',29),('credit',36.5),('grounding',47),('overview',52)]:
            (args.frames_dir/f'{name}.svg').write_text(make_svg(t))
        print(args.frames_dir)
    else:
        dest=args.output or HERE.parent/'evidence-credit.svg'
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(make_svg(args.frame))
        print(f'{dest} ({dest.stat().st_size:,} bytes; {DURATION}s)')
