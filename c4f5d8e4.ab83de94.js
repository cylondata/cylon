(window.webpackJsonp=window.webpackJsonp||[]).push([[22],{122:function(e,a,t){"use strict";t.r(a);var n=t(2),l=t(0),r=t.n(l),i=t(136),c=t(147),o=t(138),s=t(134),m=t(144),d=t(123),p=t.n(d),g=(t(124),t(662)),u=t(661),E=t(280),h=(t(125),t(656)),y=t(657),f=(t(126),[{title:r.a.createElement(r.a.Fragment,null,"Fast & Scalable"),imageUrl:"img/fast.png",description:r.a.createElement(r.a.Fragment,null,"Cylon uses OpenMPI underneath. It provides core data processing operators many times efficiently than current systems.")},{title:r.a.createElement(r.a.Fragment,null,"Designed to be Integrated"),imageUrl:"img/compatible.png",description:r.a.createElement(r.a.Fragment,null,"Cylon is designed to work across different data processing frameworks, deep learning frameworks and data formats.")}]),b=[{title:r.a.createElement(r.a.Fragment,null,"Powered by Apache Arrow"),imageUrl:"https://arrow.apache.org/img/arrow.png",description:r.a.createElement(r.a.Fragment,null,"Cylon uses Apache Arrow underneath to represent data.")},{title:r.a.createElement(r.a.Fragment,null,"We have flattened the Curve, The Learning Curve!"),imageUrl:"img/curve.png",description:r.a.createElement(r.a.Fragment,null,"Write in the language you are already familiar with, yet experience the same native performance.")}];function v(e){var a=e.imageUrl,t=e.title,n=e.description,l=e.fixedWidth,c=void 0!==l&&l,o=e.col,s=void 0===o?"col--4":o,d=Object(m.a)(a);return r.a.createElement("div",{className:Object(i.a)("col "+s,p.a.feature)},d&&r.a.createElement("div",{className:"text--left"},r.a.createElement("img",{className:c?p.a.featureImageFixedWidth:p.a.featureImage,src:d,alt:t})),r.a.createElement("h3",null,t),r.a.createElement("p",null,n))}var w=[{world:16,cpp:19.66533333,python:19.61566667,java:20.96333333},{world:32,cpp:10.198,python:10.072,java:10.654},{world:64,cpp:5.241,python:5.310333333,java:5.462333333},{world:128,cpp:3.108333333,python:3.106666667,java:3.139666667}],S=[{worldSize:"1",cylonH:"141.5",cylonS:"164.2",spark:"586.5"},{worldSize:"2",cylonH:"121.2",cylonS:"116.2",spark:"332.8"},{worldSize:"4",cylonH:"61.6",cylonS:"56.5",spark:"207.1"},{worldSize:"8",cylonH:"30.7",cylonS:"27.4",spark:"119.0"},{worldSize:"16",cylonH:"15.0",cylonS:"13.2",spark:"62.3"},{worldSize:"32",cylonH:"8.1",cylonS:"7.0",spark:"39.6"},{worldSize:"64",cylonH:"4.5",cylonS:"4.0",spark:"22.2"},{worldSize:"128",cylonH:"2.8",cylonS:"2.5",spark:"18.1"},{worldSize:"160",cylonH:"2.5",cylonS:"2.3",spark:"18.0"}],k=[{worldSize:"1",cylon:"342.537",spark:"748.567"},{worldSize:"2",cylon:"223.186",spark:"412.414"},{worldSize:"4",cylon:"108.53",spark:"239.747"},{worldSize:"8",cylon:"53.022",spark:"141.44"},{worldSize:"16",cylon:"26.676",spark:"69.525"},{worldSize:"32",cylon:"14.064",spark:"43.651"},{worldSize:"64",cylon:"8.399",spark:"23.895"},{worldSize:"128",cylon:"6.773",spark:"18.384"},{worldSize:"160",cylon:"7.261",spark:"17.572"}];a.default=function(){var e=Object(s.a)().siteConfig,a=void 0===e?{}:e;return r.a.createElement(c.a,{title:"Hello from "+a.title,description:"Description will go into a meta tag in <head />"},r.a.createElement("header",{className:Object(i.a)("hero hero--primary",p.a.heroBanner,"custom-background")},r.a.createElement("div",{className:"container header-container"},r.a.createElement("h1",{className:"hero__title"},a.title),r.a.createElement("p",{className:"hero__subtitle"},a.tagline),r.a.createElement("div",{className:p.a.buttons},r.a.createElement(o.a,{className:Object(i.a)("button button--outline button--secondary button--lg",p.a.getStarted),to:Object(m.a)("docs/")},"Get Started")),r.a.createElement("img",{src:"img/wheel.svg",className:"header-img"}))),r.a.createElement("main",null,f&&f.length>0&&r.a.createElement("section",{className:p.a.features},r.a.createElement("div",{className:"container"},r.a.createElement("div",{className:"row"},f.map((function(e,a){return r.a.createElement(v,Object(n.a)({key:a},e))}))))),b&&b.length>0&&r.a.createElement("section",{className:p.a.features},r.a.createElement("div",{className:"container"},r.a.createElement("div",{className:"row"},r.a.createElement("div",{className:Object(i.a)("col col--4",p.a.feature)},b.map((function(e,a){return r.a.createElement(v,Object(n.a)({key:a},e,{fixedWidth:!0,col:"col--12"}))}))),r.a.createElement("div",{className:Object(i.a)("col col--8",p.a.feature)},r.a.createElement(y.d,null,r.a.createElement(y.b,null,r.a.createElement(y.a,null,r.a.createElement("i",{className:"devicon-cplusplus-plain"})),r.a.createElement(y.a,null,r.a.createElement("i",{className:"devicon-java-plain"})),r.a.createElement(y.a,null,r.a.createElement("i",{className:"devicon-python-plain"}))),r.a.createElement(y.c,null,r.a.createElement(g.a,{language:"cpp",style:u.a,showLineNumbers:!0},'int main(int argc, char *argv[]) {\n  auto mpi_config = new MPIConfig();\n  auto ctx = CylonContext::InitDistributed(mpi_config);\n  std::shared_ptr<Table> table1, table2, joined;\n\n  auto read_options = CSVReadOptions().UseThreads(true);\n  Table::FromCSV(ctx, {\n      "/path/to/csv1.csv",\n      "/path/to/csv2.csv"\n  }, {table1, table2}, read_options);\n\n  auto join_config = JoinConfig::InnerJoin(0, 0);\n  table1->DistributedJoin(table2, join_config, &joined);\n  joined->Print();\n  \n  ctx->Finalize();\n  return 0;\n}')),r.a.createElement(y.c,null,r.a.createElement(g.a,{language:"java",style:u.a,showLineNumbers:!0},"public class DistributedJoinExample {\n                                                \n  public static void main(String[] args) {\n    String src1Path = args[0];\n    String src2Path = args[1];\n\n    CylonContext ctx = CylonContext.init();\n\n    Table left = Table.fromCSV(ctx, src1Path);\n    Table right = Table.fromCSV(ctx, src2Path);\n\n    JoinConfig joinConfig = new JoinConfig(0, 0);\n    Table joined = left.distributedJoin(right, joinConfig);\n    joined.print();\n    \n    ctx.finalizeCtx();\n  }\n}")))))))),r.a.createElement("div",{className:"performance-charts custom-background"},r.a.createElement("div",{className:"container performance-charts-content"},r.a.createElement("h3",null,"Written with Performance & Scalability in Mind!"),r.a.createElement(E.a,{naturalSlideWidth:100,naturalSlideHeight:400,isPlaying:!0,infinite:!0,interval:5e3,totalSlides:3},r.a.createElement(E.c,null,r.a.createElement(E.b,{index:0},r.a.createElement("h4",null,"Cross Language Performance"),r.a.createElement("p",null,"Experiment informatiom goes here"),r.a.createElement(h.g,{width:"100%",height:330},r.a.createElement(h.b,{data:w},r.a.createElement(h.a,{fill:"#00BCD4",dataKey:"cpp"}),r.a.createElement(h.a,{fill:"#4CAF50",dataKey:"python"}),r.a.createElement(h.a,{fill:"#FF5722",dataKey:"java"}),r.a.createElement(h.i,{dataKey:"world",label:"World Size"}),r.a.createElement(h.j,{label:{value:"time(s)",angle:-90,position:"insideLeft"}}),r.a.createElement(h.d,{verticalAlign:"top",height:36})))),r.a.createElement(E.b,{index:1},r.a.createElement("h4",null,"Distributed Join(String Scaling)"),r.a.createElement("p",null,"Cylon(Hash Join) vs Cylon(Sort Join) vs Spark"),r.a.createElement(h.g,{width:"100%",height:330},r.a.createElement(h.f,{data:S},r.a.createElement(h.e,{stroke:"#00BCD4",dataKey:"cylonH"}),r.a.createElement(h.e,{stroke:"#4CAF50",dataKey:"cylonS"}),r.a.createElement(h.e,{stroke:"#FF5722",dataKey:"spark"}),r.a.createElement(h.i,{dataKey:"worldSize"},r.a.createElement(h.c,{value:"World Size",offset:0,position:"insideBottom"})),r.a.createElement(h.j,{label:{value:" time(s)",angle:-90,position:"insideLeft"}}),r.a.createElement(h.d,{verticalAlign:"top",height:36}),r.a.createElement(h.h,null)))),r.a.createElement(E.b,{index:2},r.a.createElement("h4",null,"Distributed Union"),r.a.createElement(h.g,{width:"100%",height:380},r.a.createElement(h.f,{data:k},r.a.createElement(h.e,{stroke:"#00BCD4",dataKey:"cylon"}),r.a.createElement(h.e,{stroke:"#FF5722",dataKey:"spark"}),r.a.createElement(h.i,{dataKey:"worldSize"},r.a.createElement(h.c,{value:"World Size",position:"insideBottom",offset:0})),r.a.createElement(h.j,{label:{value:" time(s)",angle:-90,position:"insideLeft"}}),r.a.createElement(h.d,{verticalAlign:"top",align:"right",height:36}),r.a.createElement(h.h,null)))))))))}}}]);