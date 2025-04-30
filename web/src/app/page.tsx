// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { GithubFilled, GithubOutlined } from "@ant-design/icons";
import {
  Bird,
  Book,
  Camera,
  ChevronRight,
  Code,
  Globe,
  Home,
  Lightbulb,
  Microscope,
  Paintbrush,
  Podcast,
  ShoppingCart,
  Usb,
  User,
} from "lucide-react";
import Link from "next/link";

import { AuroraText } from "~/components/magicui/aurora-text";
import { BentoCard } from "~/components/magicui/bento-grid";
import { BentoGrid } from "~/components/magicui/bento-grid";
import { FlickeringGrid } from "~/components/magicui/flickering-grid";
import { Button } from "~/components/ui/button";

import { MultiAgentVisualization } from "./_components/multi-agent-visualization";
import { Ray } from "./_components/ray";

export default function HomePage() {
  return (
    <div className="flex flex-col items-center">
      <Header />
      <main className="container flex flex-col items-center justify-center gap-56">
        <Jumbotron />
        <CaseStudySection />
        <MultiAgentSection />
        <CoreFeatureSection />
        <JoinCommunitySection />
      </main>
      <Footer />
      <Ray />
    </div>
  );
}

function Header() {
  return (
    <header className="supports-backdrop-blur:bg-background/80 bg-background/40 sticky top-0 left-0 z-40 flex h-15 w-full flex-col items-center backdrop-blur-lg">
      <div className="container flex h-15 items-center justify-between px-3">
        <div className="text-xl font-medium">
          <span className="mr-1 text-2xl">🦌</span>
          <span>DeerFlow</span>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" asChild>
            <Link href="https://github.com/bytedance/deer-flow" target="_blank">
              <GithubOutlined />
              Star on GitHub
            </Link>
          </Button>
        </div>
      </div>
      <hr className="from-border/0 via-border/70 to-border/0 m-0 h-px w-full border-none bg-gradient-to-r" />
    </header>
  );
}

function Footer() {
  return (
    <footer className="container mt-32 flex flex-col items-center justify-center">
      <hr className="from-border/0 via-border/70 to-border/0 m-0 h-px w-full border-none bg-gradient-to-r" />
      <div className="text-muted-foreground container flex h-20 flex-col items-center justify-center text-sm">
        <p className="text-center font-serif text-lg md:text-xl">
          &quot;Originated from Open Source, give back to Open Source.&quot;
        </p>
      </div>
      <div className="text-muted-foreground container mb-8 flex flex-col items-center justify-center text-xs">
        <p>Licensed under MIT License</p>
        <p>&copy; 2025 DeepFlow</p>
      </div>
    </footer>
  );
}

function SectionHeader({
  title,
  description,
}: {
  title: React.ReactNode;
  description: React.ReactNode;
}) {
  return (
    <div className="mb-12 flex flex-col items-center justify-center gap-2">
      <h2 className="mb-4 bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-center text-5xl font-bold text-transparent">
        {title}
      </h2>
      <p className="text-muted-foreground text-center text-xl">{description}</p>
    </div>
  );
}

function Jumbotron() {
  return (
    <section className="relative flex h-[95vh] flex-col items-center justify-center pb-15">
      <FlickeringGrid
        id="deer-hero-bg"
        className={`absolute inset-0 z-0 [mask-image:radial-gradient(800px_circle_at_center,white,transparent)]`}
        squareSize={4}
        gridGap={4}
        color="#60A5FA"
        maxOpacity={0.133}
        flickerChance={0.1}
      />
      <FlickeringGrid
        id="deer-hero"
        className={`absolute inset-0 z-0 mask-[url(/images/deer-hero.svg)] mask-size-[75%] mask-center mask-no-repeat`}
        squareSize={3}
        gridGap={6}
        color="#60A5FA"
        maxOpacity={0.66}
        flickerChance={0.12}
      />
      <div className="relative z-10 flex flex-col items-center justify-center gap-12">
        <h1 className="text-center text-4xl font-bold md:text-6xl">
          <span className="bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent">
            Deep Research{" "}
          </span>
          <AuroraText>at Your Fingertips</AuroraText>
        </h1>
        <p className="max-w-4xl p-2 text-center text-sm font-light opacity-80 md:text-2xl">
          Meet <span className="font-medium">DeerFlow</span>, your ultimate Deep
          Research assistant. With powerful tools like search engines, web
          crawlers, Python and MCP services, it provides instant insights,
          comprehensive reports, or even captivating podcasts.
        </p>
        <div className="flex gap-6">
          <Button className="hidden text-lg md:flex md:w-42" size="lg" asChild>
            <Link href="/chat">
              Get Started <ChevronRight />
            </Link>
          </Button>
          <Button className="w-42 text-lg" size="lg" variant="outline" asChild>
            <Link href="https://github.com/bytedance/deer-flow" target="_blank">
              <GithubFilled />
              Learn More
            </Link>
          </Button>
        </div>
      </div>
      <div className="absolute bottom-8 flex text-xs opacity-50">
        <p>* DEER stands for Deep Exploration and Efficient Research.</p>
      </div>
    </section>
  );
}

const caseStudies = [
  {
    icon: Home,
    title: "Design a futuristic smart home for a family of 4",
    description:
      "DeepFlow generates innovative concepts and detailed blueprints for a cutting-edge smart home, incorporating AI-driven automation and sustainable energy solutions.",
  },
  {
    icon: Book,
    title: "Write a historical fiction novel set in ancient Rome",
    description:
      "DeepFlow assists in crafting compelling characters, intricate plots, and historically accurate settings to bring your ancient Roman story to life.",
  },
  {
    icon: ShoppingCart,
    title: "Create a marketing campaign for a new eco-friendly product",
    description:
      "DeepFlow develops a strategic marketing plan, including social media content, ad designs, and audience targeting strategies to promote your sustainable product effectively.",
  },
  {
    icon: Lightbulb,
    title: "Invent a groundbreaking renewable energy solution",
    description:
      "DeepFlow helps brainstorm innovative ideas and provides technical schematics for a renewable energy solution that addresses global energy challenges.",
  },
  {
    icon: Camera,
    title: "Direct a short film about climate change",
    description:
      "DeepFlow assists in developing a compelling script, visual storyboards, and production plans to create an impactful short film on environmental issues.",
  },
  {
    icon: Globe,
    title: "Plan a sustainable travel itinerary for eco-conscious tourists",
    description:
      "DeepFlow designs a detailed travel plan featuring eco-friendly accommodations, activities, and transportation options for environmentally aware travelers.",
  },
  {
    icon: Code,
    title: "Develop an AI-powered app for mental health support",
    description:
      "DeepFlow provides guidance on app design, user experience, and AI integration to create a tool that offers personalized mental health resources and support.",
  },
  {
    icon: Paintbrush,
    title: "Create an immersive art installation on human connection",
    description:
      "DeepFlow assists in conceptualizing and designing an interactive art installation that explores themes of empathy, relationships, and the power of human connection.",
  },
];

function CaseStudySection() {
  return (
    <section className="relative container hidden flex-col items-center justify-center md:flex">
      <SectionHeader
        title="Case Studies"
        description="See DeerFlow in action through replays."
      />
      <div className="grid w-3/4 grid-cols-1 gap-2 sm:w-full sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {caseStudies.map((caseStudy) => (
          <div key={caseStudy.title} className="w-full p-2">
            <BentoCard
              {...{
                Icon: caseStudy.icon,
                name: caseStudy.title,
                description: caseStudy.description,
                href: "/",
                cta: "Learn more",
                className: "w-full h-full",
              }}
            />
          </div>
        ))}
      </div>
    </section>
  );
}

function MultiAgentSection() {
  return (
    <section className="relative flex w-full flex-col items-center justify-center">
      <SectionHeader
        title="Multi-Agent Architecture"
        description="Experience the agent teamwork with our Supervisor + Handoffs design pattern."
      />
      <div className="flex h-[70vh] w-full flex-col items-center justify-center">
        <div className="h-full w-full">
          <MultiAgentVisualization />
        </div>
      </div>
    </section>
  );
}

const features = [
  {
    Icon: Microscope,
    name: "Dive Deeper and Reach Wider",
    description:
      "Unlock deeper insights with advanced tools. Our powerful search + crawling and Python tools gathers comprehensive data, delivering in-depth reports to enhance your study.",
    href: "/",
    cta: "Learn more",
    background: (
      <img alt="background" className="absolute -top-20 -right-20 opacity-60" />
    ),
    className: "lg:col-start-1 lg:col-end-2 lg:row-start-1 lg:row-end-3",
  },
  {
    Icon: User,
    name: "Human-in-the-loop",
    description:
      "Refine your research plan, or adjust focus areas all through simple natural language.",
    href: "/",
    cta: "Learn more",
    background: (
      <img alt="background" className="absolute -top-20 -right-20 opacity-60" />
    ),
    className: "lg:col-start-1 lg:col-end-2 lg:row-start-3 lg:row-end-4",
  },
  {
    Icon: Bird,
    name: "Lang Stack",
    description:
      "Build with confidence using the LangChain and LangGraph frameworks.",
    href: "/",
    cta: "Learn more",
    background: (
      <img alt="background" className="absolute -top-20 -right-20 opacity-60" />
    ),
    className: "lg:col-start-2 lg:col-end-3 lg:row-start-1 lg:row-end-2",
  },
  {
    Icon: Usb,
    name: "MCP Integrations",
    description:
      "Supercharge your research workflow and expand your toolkit with seamless MCP integrations.",
    href: "/",
    cta: "Learn more",
    background: (
      <img alt="background" className="absolute -top-20 -right-20 opacity-60" />
    ),
    className: "lg:col-start-2 lg:col-end-3 lg:row-start-2 lg:row-end-3",
  },
  {
    Icon: Podcast,
    name: "Podcast Generation",
    description:
      "Instantly generate podcasts from reports. Perfect for on-the-go learning or sharing findings effortlessly.  ",
    href: "/",
    cta: "Learn more",
    background: (
      <img alt="background" className="absolute -top-20 -right-20 opacity-60" />
    ),
    className: "lg:col-start-2 lg:col-end-3 lg:row-start-3 lg:row-end-4",
  },
];

function CoreFeatureSection() {
  return (
    <section className="relative flex w-full flex-col content-around items-center justify-center">
      <SectionHeader
        title="Core Features"
        description="Find out what makes DeerFlow effective."
      />
      <BentoGrid className="w-3/4 lg:grid-cols-2 lg:grid-rows-3">
        {features.map((feature) => (
          <BentoCard key={feature.name} {...feature} />
        ))}
      </BentoGrid>
    </section>
  );
}

function JoinCommunitySection() {
  return (
    <section className="flex w-full flex-col items-center justify-center pb-12">
      <SectionHeader
        title={
          <AuroraText colors={["#60A5FA", "#A5FA60", "#A560FA"]}>
            Join the DeerFlow Community
          </AuroraText>
        }
        description="Contribute brilliant ideas to shape the future of DeerFlow. Collaborate, innovate, and make impacts."
      />
      <Button className="text-xl" size="lg" asChild>
        <Link href="https://github.com/bytedance/deer-flow" target="_blank">
          <GithubFilled />
          Contribute Now
        </Link>
      </Button>
    </section>
  );
}
