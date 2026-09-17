"""Build three compatible lettering masters from the supplied SVGs (stdlib only)."""
import json, math, re
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent

def contours(filename):
    result = []
    for node in ET.parse(filename).getroot():
        tokens = re.findall(r'[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?', node.attrib['d'])
        i = 0; x = y = 0; last = None; points = []
        while i < len(tokens):
            if tokens[i].isalpha(): command = tokens[i]; i += 1
            kind = command.upper(); relative = command.islower()
            if kind == 'Z':
                if points: result.append(points); x,y = points[0]; points = []
                last = None
                continue
            count = {'M':2,'L':2,'H':1,'V':1,'C':6,'S':4}[kind]
            vals = list(map(float,tokens[i:i+count])); i += count
            ox,oy=x,y
            if kind in ('M','L'):
                x,y=vals; x += ox if relative else 0; y += oy if relative else 0
                if kind == 'M':
                    if points: result.append(points)
                    points=[]; command='l' if relative else 'L'
                points.append((x,y)); last=None
            elif kind in ('H','V'):
                if kind == 'H': x=vals[0]+(ox if relative else 0)
                else: y=vals[0]+(oy if relative else 0)
                points.append((x,y)); last=None
            else:
                pairs=[(vals[j]+(ox if relative else 0),vals[j+1]+(oy if relative else 0)) for j in range(0,count,2)]
                if kind == 'S': pairs.insert(0,(2*ox-last[0],2*oy-last[1]) if last else (ox,oy))
                a,b,end=pairs
                for j in range(1,33):
                    t=j/32; u=1-t
                    points.append((u**3*ox+3*u*u*t*a[0]+3*u*t*t*b[0]+t**3*end[0],u**3*oy+3*u*u*t*a[1]+3*u*t*t*b[1]+t**3*end[1]))
                x,y=end; last=b
        if points: result.append(points)
    return result

def area(p): return sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(p,p[1:]+p[:1]))/2

def resample(p,n=520):
    p=p+[p[0]]; distances=[0]
    for a,b in zip(p,p[1:]): distances.append(distances[-1]+math.dist(a,b))
    # Retain sharp original vertices as well as uniform samples, including fine tips.
    samples={distances[-1]*i/n for i in range(n)}
    for i in range(len(p)-1):
        prev=p[i-1] if i else p[-2]
        a=(p[i][0]-prev[0],p[i][1]-prev[1])
        b=(p[i+1][0]-p[i][0],p[i+1][1]-p[i][1])
        length=math.hypot(*a)*math.hypot(*b)
        if length and (a[0]*b[0]+a[1]*b[1])/length<.96: samples.add(distances[i])
    output=[]; j=0
    for d in sorted(samples):
        while j+1<len(distances)-1 and distances[j+1]<d: j+=1
        t=(d-distances[j])/max(1e-9,distances[j+1]-distances[j])
        output.append(tuple(p[j][k]*(1-t)+p[j+1][k]*t for k in (0,1)))
    return output


def rotate_near(points, landmark):
    i=min(range(len(points)),key=lambda i:math.dist(points[i],landmark))
    return points[i:]+points[:i]

def normals(points):
    result=[]
    for i,p in enumerate(points):
        a=points[(i-1)%len(points)]; b=points[(i+1)%len(points)]
        dx,dy=b[0]-a[0],b[1]-a[1]; length=math.hypot(dx,dy)
        result.append((dy/length,-dx/length) if length else (0,0))
    return result

def remove_offset_loops(points):
    # Collapse small self-intersections caused by inward offsets at pointed terminals.
    points=list(points); count=len(points)
    cross=lambda a,b: a[0]*b[1]-a[1]*b[0]
    for _ in range(80):
        found=False
        for i in range(count):
            p=points[i]; q=points[(i+1)%count]; a=(q[0]-p[0],q[1]-p[1])
            if math.hypot(*a)<1e-7: continue
            for j in range(i+2,count):
                if i==0 and j==count-1: continue
                r=points[j]; t=points[(j+1)%count]; b=(t[0]-r[0],t[1]-r[1])
                denominator=cross(a,b)
                if abs(denominator)<1e-8: continue
                delta=(r[0]-p[0],r[1]-p[1])
                u=cross(delta,b)/denominator; v=cross(delta,a)/denominator
                if not (1e-6<u<1-1e-6 and 1e-6<v<1-1e-6): continue
                hit=(p[0]+u*a[0],p[1]+u*a[1])
                inside=list(range(i+1,j+1)); outside=list(range(j+1,count))+list(range(i+1))
                loop=inside if abs(area([hit]+[points[k] for k in inside]))<abs(area([hit]+[points[k] for k in outside])) else outside
                for k in loop: points[k]=hit
                found=True; break
            if found: break
        if not found: break
    return points

def safe_inward_normals(points):
    # Taper thinning near hairlines and pointed terminals to keep the outline intact.
    result=[]
    cross=lambda a,b: a[0]*b[1]-a[1]*b[0]
    count=len(points)
    for i,(p,n) in enumerate(zip(points,normals(points))):
        direction=(-n[0],-n[1]); nearest=float('inf')
        for j,q in enumerate(points):
            if min((j-i)%count,(i-j)%count)<=2: continue
            r=points[(j+1)%count]; edge=(r[0]-q[0],r[1]-q[1])
            denominator=cross(direction,edge)
            if abs(denominator)<1e-8: continue
            delta=(q[0]-p[0],q[1]-p[1])
            t=cross(delta,edge)/denominator; u=cross(delta,direction)/denominator
            if t>.01 and 0<=u<=1: nearest=min(nearest,t)
        scale=min(1,nearest*.30/8) if math.isfinite(nearest) else 0
        result.append((n[0]*scale,n[1]*scale))
    reduced=remove_offset_loops([(p[0]-n[0]*8,p[1]-n[1]*8) for p,n in zip(points,result)])
    return [((p[0]-q[0])/8,(p[1]-q[1])/8) for p,q in zip(points,reduced)]

def match(a,b):
    # Dynamic time warping preserves outline order while pairing corresponding strokes.
    n=len(a); m=len(b)
    def normalized(p):
        xs=[v[0] for v in p]; ys=[v[1] for v in p]
        mx=(min(xs)+max(xs))/2; my=(min(ys)+max(ys))/2
        return [((x-mx)/(max(xs)-min(xs)),(y-my)/(max(ys)-min(ys))) for x,y in p]
    na,nb=normalized(a),normalized(b)
    costs=[[float('inf')]*(m+1) for _ in range(n+1)]; costs[0][0]=0
    trace={}
    for i in range(1,n+1):
        for j in range(1,m+1):
            absolute=((a[i-1][0]-b[j-1][0])/400)**2+((a[i-1][1]-b[j-1][1])/400)**2
            relative=sum((na[i-1][k]-nb[j-1][k])**2 for k in (0,1))
            choices=[(costs[i-1][j-1],(i-1,j-1)),(costs[i-1][j]+.003,(i-1,j)),(costs[i][j-1]+.003,(i,j-1))]
            cost,prev=min(choices)
            costs[i][j]=cost+absolute*.7+relative*.3
            trace[i,j]=prev
    indices=[]; i,j=n,m
    while i and j:
        indices.append((i-1,j-1)); i,j=trace[i,j]
    indices.reverse()
    return indices

def master_vectors(points):
    outward=remove_offset_loops([(p[0]+n[0]*12,p[1]+n[1]*12) for p,n in zip(points,normals(points))])
    inward=safe_inward_normals(points)
    return [[*p,q[0]-p[0],q[1]-p[1],-n[0]*8,-n[1]*8] for p,q,n in zip(points,outward,inward)]

def compatible_masters(a,m,b):
    # Join both correspondences through the same middle outline. Each source point
    # survives, so crossing the middle never switches to a different point mapping.
    am=match(a,m); mb=match(m,b)
    left=[[] for _ in m]; right=[[] for _ in m]
    for ai,mi in am: left[mi].append(ai)
    for mi,bi in mb: right[mi].append(bi)
    av,mv,bv=map(master_vectors,(a,m,b))
    def at(values,position):
        i=int(position); j=min(i+1,len(values)-1); t=position-i
        return [u+(v-u)*t for u,v in zip(values[i],values[j])]
    result=[]
    for mi in range(len(m)):
        count=max(len(left[mi]),len(right[mi]))
        for k in range(count):
            t=k/(count-1) if count>1 else 0
            ai=left[mi][0]+(left[mi][-1]-left[mi][0])*t
            bi=right[mi][0]+(right[mi][-1]-right[mi][0])*t
            result.append([round(v,4) for v in [*at(av,ai),*mv[mi],*at(bv,bi)]])
    return result

def build():
    light=contours(ROOT/'assets/lettering-light-sans.svg')
    middle=contours(ROOT/'assets/lettering-medium.svg')
    bold=contours(ROOT/'assets/lettering-bold-serif.svg')
    pairs=[(0,0,(433,408),(452,392),(452,392)),
           (2,1,(514,372),(528,262),(549,173)),
           (1,2,(514,573),(525,549),(525,549)),
           (4,3,(832,495),(869,487),(884,472)),
           (3,4,(869,549),(923,517),(1026,550))]
    data=[]
    for ai,bi,aa,ma,ba in pairs:
        shapes=[]
        for points,anchor in ((light[ai],aa),(middle[bi],ma),(bold[bi],ba)):
            if area(points)<0: points=list(reversed(points))
            shapes.append(resample(rotate_near(points,anchor)))
        data.append(compatible_masters(*shapes))
    master_paths={}
    for name in ('light-sans','medium','bold-serif'):
        master_paths[name]=[node.attrib['d'] for node in ET.parse(ROOT/f'assets/lettering-{name}.svg').getroot()]
    payload=json.dumps({'outlines':data,'masters':master_paths},separators=(',',':'))
    page=ROOT/'index.html'
    if page.stat().st_size:
        text=page.read_text()
        text=re.sub(r'(?<=<script id="lettering-data" type="application/json">).*?(?=</script>)',lambda _:payload,text,flags=re.S)
        page.write_text(text)
    else:
        (ROOT/'assets/morph-data.json').write_text(payload)
    print('Matched',sum(map(len,data)),'points across five components.')

if __name__ == '__main__': build()
