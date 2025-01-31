"use client";
import React from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import Image from "next/image";

const transition = {
  type: "spring",
  mass: 0.5,
  damping: 11.5,
  stiffness: 100,
  restDelta: 0.001,
  restSpeed: 0.001,
};

export const MenuItem = ({
  setActive,
  active,
  item,
  children,
  className
}) => {
  return (
    <div onMouseEnter={() => setActive(item)} className="relative">
      <motion.p
        transition={{ duration: 0.3 }}
        className={`cursor-pointer text-white hover:opacity-[0.9] text-sm font-medium ${className}`}>
        {item}
      </motion.p>
      {active !== null && (
        <motion.div
          initial={{ opacity: 0, scale: 0.85, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={transition}>
          {active === item && (
            <div className="absolute top-[calc(100%_+_1.2rem)] left-1/2 transform -translate-x-1/2 pt-4">
              <motion.div
                transition={transition}
                layoutId="active"
                className="bg-black backdrop-blur-sm rounded-2xl overflow-hidden border border-white/[0.2] shadow-xl">
                <motion.div
                  layout
                  className="w-max h-full p-4">
                  {children}
                </motion.div>
              </motion.div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
};

export const Menu = ({
  setActive,
  children
}) => {
  return (
    <nav
      onMouseLeave={() => setActive(null)}
      className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50 w-[40%] min-w-[800px] bg-black border border-white/[0.2] rounded-full px-8 py-4"
    >
      <div className="flex justify-between items-center">
        {children}
      </div>
    </nav>
  );
};

export const NavSection = ({
  title,
  description,
  href,
  src
}) => {
  const isHome = title === "Home";
  return (
    <Link href={href} className="flex space-x-2">
      {src && (
        <Image
          src={src}
          width={140}
          height={70}
          alt={title}
          className="flex-shrink-0 rounded-md shadow-2xl"
        />
      )}
      <div className="text-center">
        <h4 className={`${isHome ? 'text-2xl' : 'text-xl'} font-bold mb-1 text-white`}>
          {title}
        </h4>
        {description && (
          <p className="text-neutral-300 text-sm max-w-[10rem]">
            {description}
          </p>
        )}
      </div>
    </Link>
  );
};

export const HoveredLink = ({
  children,
  ...rest
}) => {
  return (
    (<Link
      {...rest}
      className="text-neutral-200 hover:text-white">
      {children}
    </Link>)
  );
};
