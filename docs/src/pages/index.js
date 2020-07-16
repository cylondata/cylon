import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';
import "./custom.css";
import {CodeBlock, monoBlue} from "react-code-blocks";

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
        title: <>Designed to be integrated</>,
        imageUrl: 'https://uucss.org/wp-content/uploads/2016/10/operations-council.png',
        description: (
            <>
                Cylon is designed to work accross different data processing frameworks, deep learning frameworks and
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

function Home() {
    const context = useDocusaurusContext();
    const {siteConfig = {}} = context;
    return (
        <Layout
            title={`Hello from ${siteConfig.title}`}
            description="Description will go into a meta tag in <head />">
            <header className={clsx('hero hero--primary', styles.heroBanner,'custom-background')}>
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
                    <img src="img/wheel.png" className="header-img"/>
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
        </Layout>
    );
}

export default Home;
