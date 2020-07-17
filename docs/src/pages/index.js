import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';
import "./custom.css";
import {CodeBlock, monoBlue} from "react-code-blocks";
import {CarouselProvider, Slide, Slider} from 'pure-react-carousel';
import 'pure-react-carousel/dist/react-carousel.es.css';
import {Bar, BarChart, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis} from 'recharts';


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
        imageUrl: 'https://fuelcycle.com/wp-content/uploads/2018/12/Fast-Flexible-Connected-01-2-310x300.png',
        description: (
            <>
                Cylon uses OpenMPI underneath. It provides core data processing operators many times efficiently than
                current systems.
            </>
        ),
    },
    {
        title: <>Designed to be Integrated</>,
        imageUrl: 'https://uucss.org/wp-content/uploads/2016/10/operations-council.png',
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
];

function Feature({imageUrl, title, description, fixedWidth = false}) {
    const imgUrl = useBaseUrl(imageUrl);
    return (
        <div className={clsx('col col--4', styles.feature)}>
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


const joinData = [{"worldSize": "1", "cylonH": "0.8025", "cylonS": "0.88675", "spark": "7.338"}, {
    "worldSize": "2",
    "cylonH": "1.6715",
    "cylonS": "1.604875",
    "spark": "7.73"
}, {"worldSize": "4", "cylonH": "1.748875", "cylonS": "1.637625", "spark": "9.4625"}, {
    "worldSize": "8",
    "cylonH": "1.82803125",
    "cylonS": "1.64871875",
    "spark": "11.148"
}, {"worldSize": "16", "cylonH": "1.889375", "cylonS": "1.6979375", "spark": "11.505"}, {
    "worldSize": "32",
    "cylonH": "2.256546875",
    "cylonS": "1.978664063",
    "spark": "12.5295"
}, {"worldSize": "64", "cylonH": "2.890671875", "cylonS": "2.53996875", "spark": "15.22"}, {
    "worldSize": "128",
    "cylonH": "3.562925781",
    "cylonS": "3.307162109",
    "spark": "21.5975"
}, {"worldSize": "160", "cylonH": "3.976823438", "cylonS": "3.821015625", "spark": "23.1185"}]

const unionData = [{
    "worldSize": "1",
    "cylon": "2.6275",
    "spark": "6.7465"
}, {"worldSize": "2", "cylon": "4.156", "spark": "7.398"}, {
    "worldSize": "4",
    "cylon": "4.1535",
    "spark": "8.671"
}, {"worldSize": "8", "cylon": "4.198", "spark": "13.5205"}, {
    "worldSize": "16",
    "cylon": "4.2725",
    "spark": "13.2175"
}, {"worldSize": "32", "cylon": "4.7045", "spark": "13.9885"}, {
    "worldSize": "64",
    "cylon": "5.6175",
    "spark": "16.742"
}, {"worldSize": "128", "cylon": "8.067", "spark": "21.227"}, {
    "worldSize": "160",
    "cylon": "9.4145",
    "spark": "23.647"
}];

function Home() {
    const context = useDocusaurusContext();
    const {siteConfig = {}} = context;
    return (
        <Layout
            title={`Hello from ${siteConfig.title}`}
            description="Description will go into a meta tag in <head />">
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
                                {features2.map((props, idx) => (
                                    <Feature key={idx} {...props} fixedWidth={true}/>
                                ))}

                                <div className={clsx('col col--8', styles.feature)}>
                                    <CodeBlock
                                        text={`int main(int argc, char *argv[]) {
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  std::shared_ptr<cylon::Table> first_table, second_table, joined;
  auto read_options = cylon::io::config::CSVReadOptions();
  auto status = cylon::Table::FromCSV(ctx, argv[1], first_table, read_options);
  status = cylon::Table::FromCSV(ctx, argv[2], second_table, read_options);
  status = first_table->DistributedJoin(second_table,
                                        cylon::join::config::JoinConfig::InnerJoin(0, 0),
                                        &joined);
  ctx->Finalize();
  return 0;
}`}
                                        language={"cpp"}
                                        theme={monoBlue}
                                        showLineNumbers={true}
                                        wrapLines
                                    />
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
                                <p>Experiment informatiom goes here</p>
                                <ResponsiveContainer width="100%" height={350}>
                                    <BarChart data={languageData}>
                                        <Bar fill="#00BCD4" dataKey="cpp"/>
                                        <Bar fill="#4CAF50" dataKey="python"/>
                                        <Bar fill="#FF5722" dataKey="java"/>
                                        <XAxis dataKey="world" label="World Size"/>
                                        <YAxis label={{value: "time(s)", angle: -90}}/>
                                        <Legend verticalAlign="top" height={36}/>
                                        {/*<Tooltip/>*/}
                                    </BarChart>
                                </ResponsiveContainer>
                            </Slide>
                            <Slide index={1}>
                                <h4>Distributed Join</h4>
                                <p>Experiment informatiom goes here</p>
                                <ResponsiveContainer width="100%" height={350}>
                                    <LineChart data={joinData}>
                                        <Line stroke="#00BCD4" dataKey="cylonH"/>
                                        <Line stroke="#4CAF50" dataKey="cylonS"/>
                                        <Line stroke="#FF5722" dataKey="spark"/>
                                        <XAxis dataKey="worldSize" label=" World Size"/>
                                        <YAxis label={{value: " time(s)", angle: -90}}/>
                                        <Legend verticalAlign="top" height={36}/>
                                        <Tooltip/>
                                    </LineChart>
                                </ResponsiveContainer>
                            </Slide>
                            <Slide index={2}>
                                <h4>Distributed Union</h4>
                                <p>Experiment informatiom goes here</p>
                                <ResponsiveContainer width="100%" height={350}>
                                    <LineChart data={unionData}>
                                        <Line stroke="#00BCD4" dataKey="cylon"/>
                                        <Line stroke="#FF5722" dataKey="spark"/>
                                        <XAxis dataKey="worldSize" label=" World Size"/>
                                        <YAxis label={{value: " time(s)", angle: -90}}/>
                                        <Legend verticalAlign="top" align="right" height={36}/>
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
