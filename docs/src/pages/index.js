import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';
import "./custom.css";
import SyntaxHighlighter from 'react-syntax-highlighter';
import {docco} from 'react-syntax-highlighter/dist/esm/styles/hljs';
import {CarouselProvider, Slide, Slider} from 'pure-react-carousel';
import 'pure-react-carousel/dist/react-carousel.es.css';
import {Bar, BarChart, Label, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis} from 'recharts';
import {Tab, TabList, TabPanel, Tabs} from 'react-tabs';
import 'react-tabs/style/react-tabs.css';
import Head from "@docusaurus/Head";


const features = [
    // {
    //     title: <>Powered by Apache Arrow</>,
    //     imageUrl: 'https://arrow.apache.org/img/arrow.png',
    //     description: (
    //         <>
    //             Cylon uses Apache Arrow underneath to represent data.
    //         </>
    //     ),
    // },
    {
        title: <>Fast & Scalable</>,
        imageUrl: 'img/fast.png',
        description: (
            <>
                Cylon uses OpenMPI underneath. It provides core data processing operators many times efficiently than
                current systems.
            </>
        ),
    },
    {
        title: <>Designed to be Integrated</>,
        imageUrl: 'img/compatible.png',
        description: (
            <>
                Cylon is designed to work across different data processing frameworks, deep learning frameworks and
                data formats.
            </>
        ),
    }
];

const features2 = [
    {
        title: <>Powered by Apache Arrow</>,
        imageUrl: 'https://arrow.apache.org/img/arrow.png',
        description: (
            <>
                Cylon uses Apache Arrow underneath to represent data.
            </>
        ),
    },
    {
        title: <>BYOL, Bring Your Own Language!</>,
        imageUrl: 'img/byol.png',
        description: (
            <>
                Write in the language you are already familiar with, yet experience the same native performance.
            </>
        ),
    },
];

function Feature({imageUrl, title, description, fixedWidth = false, col = 'col--4', style = {}}) {
    const imgUrl = useBaseUrl(imageUrl);
    return (
        <div className={clsx('col ' + col, styles.feature)} style={style}>
            {imgUrl && (
                <div className="text--left">
                    <img className={fixedWidth ? styles.featureImageFixedWidth : styles.featureImage} src={imgUrl}
                         alt={title}/>
                </div>
            )}
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
    );
}

const languageData = [
    {world: 16, cpp: 19.66533333, python: 19.61566667, java: 20.96333333},
    {world: 32, cpp: 10.198, python: 10.072, java: 10.654},
    {world: 64, cpp: 5.241, python: 5.310333333, java: 5.462333333},
    {world: 128, cpp: 3.108333333, python: 3.106666667, java: 3.139666667},
];


const joinData = [{"worldSize": "1", "cylonH": "141.5", "cylonS": "164.2", "spark": "586.5"}, {
    "worldSize": "2",
    "cylonH": "121.2",
    "cylonS": "116.2",
    "spark": "332.8"
}, {"worldSize": "4", "cylonH": "61.6", "cylonS": "56.5", "spark": "207.1"}, {
    "worldSize": "8",
    "cylonH": "30.7",
    "cylonS": "27.4",
    "spark": "119.0"
}, {"worldSize": "16", "cylonH": "15.0", "cylonS": "13.2", "spark": "62.3"}, {
    "worldSize": "32",
    "cylonH": "8.1",
    "cylonS": "7.0",
    "spark": "39.6"
}, {"worldSize": "64", "cylonH": "4.5", "cylonS": "4.0", "spark": "22.2"}, {
    "worldSize": "128",
    "cylonH": "2.8",
    "cylonS": "2.5",
    "spark": "18.1"
}, {"worldSize": "160", "cylonH": "2.5", "cylonS": "2.3", "spark": "18.0"}]

const unionData = [{"worldSize": "1", "cylon": "342.537", "spark": "748.567"}, {
    "worldSize": "2",
    "cylon": "223.186",
    "spark": "412.414"
}, {"worldSize": "4", "cylon": "108.53", "spark": "239.747"}, {
    "worldSize": "8",
    "cylon": "53.022",
    "spark": "141.44"
}, {"worldSize": "16", "cylon": "26.676", "spark": "69.525"}, {
    "worldSize": "32",
    "cylon": "14.064",
    "spark": "43.651"
}, {"worldSize": "64", "cylon": "8.399", "spark": "23.895"}, {
    "worldSize": "128",
    "cylon": "6.773",
    "spark": "18.384"
}, {"worldSize": "160", "cylon": "7.261", "spark": "17.572"}];

function Home() {
    const context = useDocusaurusContext();
    const {siteConfig = {}} = context;


    return (
        <Layout
            title={`${siteConfig.title}`}
            description={`${siteConfig.tagline}`}>
            <>
                <Head>
                    <meta property="og:title" content={`${siteConfig.title}`}/>
                    <meta property="og:description" content={`${siteConfig.tagline}`}/>
                    <meta property="og:image" content="https://cylondata.org/img/cylon_twitter.png"/>
                    <meta property="og:url" content="https://cylondata.org"/>
                    <meta property="twitter:title" content={`${siteConfig.title}`}/>
                    <meta property="twitter:description" content={`${siteConfig.tagline}`}/>
                    <meta property="twitter:image" content="https://cylondata.org/img/cylon_twitter.png"/>
                    <meta name="twitter:card" content="summary_large_image"/>
                </Head>
            </>
            <header className={clsx('hero hero--primary', styles.heroBanner, 'custom-background')}>
                <div className="container header-container">
                    <h1 className="hero__title">{siteConfig.title}</h1>
                    <p className="hero__subtitle">{siteConfig.tagline}</p>
                    <div className={styles.buttons}>
                        <Link
                            className={clsx(
                                'button button--outline button--secondary button--lg',
                                styles.getStarted,
                            )}
                            to={useBaseUrl('docs/')}>
                            Get Started
                        </Link>
                    </div>
                    <img src="img/wheel.svg" className="header-img"/>
                </div>
            </header>
            <main>
                {features && features.length > 0 && (
                    <section className={styles.features}>
                        <div className="container">
                            <div className="row">
                                {features.map((props, idx) => (
                                    <Feature key={idx} {...props} />
                                ))}
                            </div>
                        </div>
                    </section>
                )}
                {features2 && features2.length > 0 && (
                    <section className={styles.features}>
                        <div className="container">
                            <div className="row">
                                <div className={clsx('col col--4', styles.feature)}>
                                    {features2.map((props, idx) => (
                                        <Feature key={idx} {...props} fixedWidth={true} col="col--12"
                                                 style={{marginTop: idx === 1 ? "4rem" : "0", paddingLeft: 0}}/>
                                    ))}
                                </div>
                                <div className={clsx('col col--8', styles.feature)}>
                                    <Tabs>
                                        <TabList>
                                            <Tab><i className="devicon-cplusplus-plain"/></Tab>
                                            <Tab><i className="devicon-java-plain"/></Tab>
                                            <Tab><i className="devicon-python-plain"/></Tab>
                                        </TabList>

                                        <TabPanel>
                                            <SyntaxHighlighter language="cpp" style={docco} showLineNumbers={true}>
                                                {`int main(int argc, char *argv[]) {

  auto mpi_config = new MPIConfig();
  auto ctx = CylonContext::InitDistributed(mpi_config);
  std::shared_ptr<Table> table1, table2, joined;

  auto read_options = CSVReadOptions().UseThreads(true);
  Table::FromCSV(ctx, {
      "/path/to/csv1.csv",
      "/path/to/csv2.csv"
  }, {table1, table2}, read_options);

  auto join_config = JoinConfig::InnerJoin(0, 0);
  table1->DistributedJoin(table2, join_config, &joined);
  
  joined->Print();
  
  ctx->Finalize();
  return 0;
}`}
                                            </SyntaxHighlighter>
                                        </TabPanel>
                                        <TabPanel>
                                            <SyntaxHighlighter language="java" style={docco} showLineNumbers={true}>
                                                {`import org.cylondata.cylon.CylonContext;
import org.cylondata.cylon.Table;
import org.cylondata.cylon.ops.JoinConfig;

public class DistributedJoinExample {
                                                
  public static void main(String[] args) {
  
    CylonContext ctx = CylonContext.init();

    Table left = Table.fromCSV(ctx, "/tmp/csv1.csv");
    Table right = Table.fromCSV(ctx, "/tmp/csv2.csv");

    Table joined = left.distributedJoin(right, new JoinConfig(0, 0));
    
    joined.print();    
    
    ctx.finalizeCtx();
  }
}`}
                                            </SyntaxHighlighter>
                                        </TabPanel>
                                        <TabPanel>
                                            <SyntaxHighlighter language="python" style={docco} showLineNumbers={true}>
                                                {`from pycylon.data.table import csv_reader
from pycylon.data.table import Table
from pycylon.ctx.context import CylonContext

ctx: CylonContext = CylonContext("mpi")

tb1: Table = csv_reader.read(ctx, '/tmp/csv1.csv', ',')
tb2: Table = csv_reader.read(ctx, '/tmp/csv2.csv', ',')

configs = {'join_type':'left', 'algorithm':'hash', 
                'left_col':0, 'right_col':0}
                
tb3: Table = tb1.distributed_join(ctx, table=tb2, 
        join_type=configs['join_type'], 
        algorithm=configs['algorithm'],
        left_col=configs['left_col'], 
        right_col=configs['right_col'])
        
tb3.show()
ctx.finalize()`}
                                            </SyntaxHighlighter>
                                        </TabPanel>
                                    </Tabs>
                                </div>
                            </div>
                        </div>
                    </section>
                )}
            </main>
            <div className="performance-charts custom-background">
                <div className="container performance-charts-content">
                    <h3>Written with Performance & Scalability in Mind!</h3>
                    <CarouselProvider
                        naturalSlideWidth={100}
                        naturalSlideHeight={400}
                        isPlaying={true}
                        infinite={true}
                        interval={5000}
                        totalSlides={3}>
                        <Slider>
                            <Slide index={0}>
                                <h4>Cross Language Performance</h4>
                                <p>Join performance with C++, Java and Python</p>
                                <ResponsiveContainer width="100%" height={330}>
                                    <BarChart data={languageData}>
                                        <Bar fill="#00BCD4" dataKey="cpp"/>
                                        <Bar fill="#4CAF50" dataKey="python"/>
                                        <Bar fill="#FF5722" dataKey="java"/>
                                        <XAxis dataKey="world" label="World Size"/>
                                        <YAxis label={{value: "time(s)", angle: -90, position: 'insideLeft'}}/>
                                        <Legend verticalAlign="top" height={36}/>
                                        {/*<Tooltip/>*/}
                                    </BarChart>
                                </ResponsiveContainer>
                            </Slide>
                            <Slide index={1}>
                                <h4>Distributed Join(Strong Scaling)</h4>
                                <p>Cylon(Hash Join) vs Cylon(Sort Join) vs Spark</p>
                                <ResponsiveContainer width="100%" height={330}>
                                    <LineChart data={joinData}>
                                        <Line stroke="#00BCD4" dataKey="cylonH"/>
                                        <Line stroke="#4CAF50" dataKey="cylonS"/>
                                        <Line stroke="#FF5722" dataKey="spark"/>
                                        <XAxis dataKey="worldSize">
                                            <Label value="World Size" offset={0} position="insideBottom"/>
                                        </XAxis>
                                        <YAxis label={{value: " time(s)", angle: -90, position: 'insideLeft'}}
                                               domain={[1, 'dataMax']}
                                               scale="log"/>
                                        <Legend verticalAlign="top" align="right" height={36}/>
                                        <Tooltip/>
                                    </LineChart>
                                </ResponsiveContainer>
                            </Slide>
                            <Slide index={2}>
                                <h4>Distributed Union</h4>
                                <ResponsiveContainer width="100%" height={380}>
                                    <LineChart data={unionData}>
                                        <Line stroke="#00BCD4" dataKey="cylon"/>
                                        <Line stroke="#FF5722" dataKey="spark"/>
                                        <XAxis dataKey="worldSize">
                                            <Label value="World Size" position="insideBottom" offset={0}/>
                                        </XAxis>
                                        <YAxis label={{value: " time(s)", angle: -90, position: 'insideLeft'}}
                                               domain={[1, 'dataMax']}
                                               scale="log"/>
                                        <Legend verticalAlign="top" height={36}/>
                                        <Tooltip/>
                                    </LineChart>
                                </ResponsiveContainer>
                            </Slide>
                        </Slider>
                    </CarouselProvider>
                </div>
            </div>
        </Layout>
    )
        ;
}

// JSON.stringify(str.split("\n").map(line=>{
//     l = line.split("\t")
//     return {worldSize:l[0],cylon:l[1], spark:l[2]}
// }))

export default Home;
