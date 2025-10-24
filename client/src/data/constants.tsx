import { Sparkles, Target, Zap, Users, TrendingUp } from "lucide-react";

export const presets = [
    {
      id: 'startup-idea',
      title: 'Startup Idea Generation',
      description: 'Generate innovative business ideas for new ventures',
      icon: <TrendingUp className="h-6 w-6" />,
      purpose: 'Create innovative and scalable startup ideas',
      schema: {
        audience: 'entrepreneurs and investors',
        domain: 'business and technology',
        tone: 'innovative and ambitious',
        constraints: 'market-viable and fundable'
      },
    },
    {
      id: 'product-feature',
      title: 'Product Feature Ideas',
      description: 'Brainstorm new features for existing products',
      icon: <Sparkles className="h-6 w-6" />,
      purpose: 'Develop compelling product features that enhance user experience',
      schema: {
        audience: 'product users and stakeholders',
        domain: 'product development',
        tone: 'user-focused and practical',
        constraints: 'technically feasible and user-friendly'
      },
      
    },
    {
      id: 'marketing-campaign',
      title: 'Marketing Campaign',
      description: 'Creative marketing and promotional strategies',
      icon: <Target className="h-6 w-6" />,
      purpose: 'Create engaging marketing campaigns that drive brand awareness',
      schema: {
        audience: 'target customers and prospects',
        domain: 'marketing and advertising',
        tone: 'creative and compelling',
        constraints: 'budget-conscious and measurable'
      },
      
    },
    {
      id: 'process-improvement',
      title: 'Process Improvement',
      description: 'Optimize workflows and operational efficiency',
      icon: <Zap className="h-6 w-6" />,
      purpose: 'Streamline processes to increase efficiency and reduce waste',
      schema: {
        audience: 'team members and stakeholders',
        domain: 'operations and management',
        tone: 'practical and results-oriented',
        constraints: 'cost-effective and implementable'
      },
      
    },
    {
      id: 'team-building',
      title: 'Team Building Activities',
      description: 'Foster collaboration and team spirit',
      icon: <Users className="h-6 w-6" />,
      purpose: 'Create engaging team building activities that strengthen collaboration',
      schema: {
        audience: 'team members and colleagues',
        domain: 'human resources and culture',
        tone: 'engaging and inclusive',
        constraints: 'budget-friendly and accessible'
      },
    }
  ];
