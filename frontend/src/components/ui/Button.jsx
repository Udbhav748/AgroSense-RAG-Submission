import { forwardRef, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Loader2 } from 'lucide-react'

const VARIANTS = {
  primary: 'btn-primary',
  secondary: 'btn-secondary',
  ghost: 'btn-ghost',
  danger: 'btn bg-danger/90 text-white shadow-soft hover:bg-danger',
}

const SIZES = {
  sm: 'h-9 px-3.5 text-xs',
  md: 'h-10 px-4.5 text-sm',
  lg: 'h-12 px-6 text-base',
}

const Button = forwardRef(function Button(
  {
    as = 'button',
    variant = 'primary',
    size = 'md',
    icon: Icon,
    iconPosition = 'left',
    loading = false,
    disabled = false,
    className = '',
    children,
    ...props
  },
  ref
) {
  // Wrapping `as` (button, react-router's Link, ...) with motion() lets
  // Button render as a link while keeping hover/tap micro-interactions.
  // Memoized so it isn't recreated (and the node remounted) every render.
  const MotionComponent = useMemo(() => motion(as), [as])
  const isDisabled = disabled || loading

  return (
    <MotionComponent
      ref={ref}
      whileHover={{ scale: isDisabled ? 1 : 1.015 }}
      whileTap={{ scale: isDisabled ? 1 : 0.97 }}
      transition={{ duration: 0.15, ease: 'easeOut' }}
      className={`${VARIANTS[variant]} ${SIZES[size]} ${className}`}
      disabled={as === 'button' ? isDisabled : undefined}
      aria-disabled={isDisabled}
      {...props}
    >
      {loading ? (
        <Loader2 size={16} className="animate-spin" />
      ) : (
        Icon && iconPosition === 'left' && <Icon size={16} strokeWidth={2.25} />
      )}
      {children}
      {!loading && Icon && iconPosition === 'right' && <Icon size={16} strokeWidth={2.25} />}
    </MotionComponent>
  )
})

export default Button
