module.exports = {
    title: 'Cylon',
    tagline: 'Data Engineering Everywhere!',
    url: 'https://cylondata.org',
    baseUrl: '/',
    favicon: 'img/favicon.ico',
    organizationName: 'CylonData', // Usually your GitHub org/user name.
    projectName: 'cylon', // Usually your repo name.
    themeConfig: {
        googleAnalytics: {
            trackingID: 'UA-173169112-1'
        },
        navbar: {
            logo: {
                alt: 'Cylon',
                src: 'img/logo.png',
            },
            items: [
                {
                    to: 'docs/',
                    activeBasePath: 'docs',
                    label: 'Docs',
                    position: 'left',
                },
                {
                    href: 'https://github.com/cylondata/cylon',
                    label: 'GitHub',
                    position: 'right',
                },
            ],
        },
        footer: {
            style: 'dark',
            links: [
                {
                    title: 'Docs',
                    items: [
                        {
                            label: 'Docs',
                            to: 'docs/',
                        },
                    ],
                },
                {
                    title: 'Community',
                    items: [
                        {
                            label: 'Mailing List',
                            href: 'mailto:cylondata@googlegroups.com',
                        },
                    ],
                },
                {
                    title: 'More',
                    items: [
                        {
                            label: 'GitHub',
                            href: 'https://github.com/cylondata/cylon',
                        },
                    ],
                },
            ],
        },
        colorMode: {
            defaultMode: 'light',
            disableSwitch: true,
        }
    },
    presets: [
        [
            '@docusaurus/preset-classic',
            {
                docs: {
                    // It is recommended to set document id as docs home page (`docs/` path).
                    homePageId: 'compile',
                    sidebarPath: require.resolve('./sidebars.js'),
                    editUrl:
                        'https://github.com/cylondata/cylon/edit/master/docs/',
                },
                blog: {
                    showReadingTime: true,
                    editUrl:
                        'https://github.com/cylondata/cylon/edit/master/blog/',
                },
                theme: {
                    customCss: require.resolve('./src/css/custom.css'),
                },
            },
        ],
    ],
    stylesheets: [
        'https://cdn.jsdelivr.net/gh/konpa/devicon@master/devicon.min.css',
    ],
    customFields: {
        twitterImage: "img/wheel.png",
    }
};
