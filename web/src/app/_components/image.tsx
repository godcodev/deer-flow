import { memo, useCallback, useEffect, useState } from "react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { cn } from "~/lib/utils";

function Image({
  className,
  imageClassName,
  imageTransition,
  src,
  alt,
  fallback = null,
}: {
  className?: string;
  imageClassName?: string;
  imageTransition?: boolean;
  src: string;
  alt: string;
  fallback?: React.ReactNode;
}) {
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState(false);

  useEffect(() => {
    setIsError(false);
    setIsLoading(true);
  }, [src]);

  const handleLoad = useCallback(() => {
    setIsError(false);
    setIsLoading(false);
  }, []);
  const handleError = useCallback(
    (e: React.SyntheticEvent<HTMLImageElement>) => {
      e.currentTarget.style.display = "none";
      console.warn(`Markdown: Image "${e.currentTarget.src}" failed to load`);
      setIsError(true);
    },
    [],
  );
  return (
    <span className={cn("block w-fit overflow-hidden", className)}>
      {isError ? (
        fallback
      ) : (
        <Tooltip>
          <TooltipTrigger asChild>
            <img
              className={cn(
                "size-full object-cover",
                imageTransition && "transition-all duration-200 ease-out",
                imageClassName,
              )}
              src={src}
              alt={alt}
              onLoad={handleLoad}
              onError={handleError}
            />
          </TooltipTrigger>
          <TooltipContent>
            <p className="max-w-64 text-sm">{alt ?? "No caption"}</p>
          </TooltipContent>
        </Tooltip>
      )}
    </span>
  );
}

export default memo(Image);
