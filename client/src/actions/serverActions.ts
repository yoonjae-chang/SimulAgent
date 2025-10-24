"use server"

import { createClient } from "@/lib/supabase/server";
import { prompts } from "@/prompts/prompts";

// Type definitions
export type Idea = {
    idea: string;
    description: string;
}

export type IdeaWithEvaluation = {
    idea: string;
    description: string;
    evaluation: string;
}

export type IdeaRanking = {
    idea: string;
    description: string;
    ranking: string;
}

export type IdeaSchema = {
    purpose: string;
    context: string;
    criteria: string[];
    constraints: string[];
    successful_ideas?: string[];
    past_ideas?: string[];
}

// Helper function to call the chat completion model
async function callChatCompletion(
    systemPrompt: string,
    userPrompt: string,
    model: string = "gpt-4o-mini",
    temperature: number = 1,
    expectArray: boolean = false
) {
    const supabase = await createClient();

    const { data: userData, error: userError } = await supabase.auth.getUser();
    if (userError) {
        throw new Error(userError.message);
    }
    if (!userData.user) {
        throw new Error('User not found');
    }

    console.log('Making chat completion request:', { model, temperature });

    for (let i = 0; i < 3; i++) {
        try {
            const response = await supabase.functions.invoke('chatCompletionModel', {
                body: {
                    systemPrompt,
                    userPrompt,
                    model,
                    temperature,
                    expectArray,
                }
            });

            console.log(`Attempt ${i + 1} response:`, {
                error: response.error,
                success: response.data?.success,
                dataType: typeof response.data?.data,
                dataLength: response.data?.data?.length
            });

            if (response.error) {
                console.error(`Attempt ${i + 1} failed:`, response.error);
                continue;
            }

            if (response.data?.success && response.data?.data) {
                try {
                    const parsed = JSON.parse(response.data.data);
                    console.log(`Attempt ${i + 1} parsed successfully:`, {
                        type: typeof parsed,
                        isArray: Array.isArray(parsed),
                        keys: typeof parsed === 'object' ? Object.keys(parsed) : 'not object'
                    });
                    return parsed;
                } catch (parseError) {
                    console.error(`Attempt ${i + 1} JSON parse failed:`, parseError);
                    console.error('Raw data:', response.data.data);
                    continue;
                }
            } else {
                console.error(`Attempt ${i + 1} unsuccessful:`, response.data);
                continue;
            }
        } catch (requestError) {
            console.error(`Attempt ${i + 1} request failed:`, requestError);
            continue;
        }
    }
    
    throw new Error('Failed to get valid response after 3 attempts');
}

/**
 * Generate initial schema based on user's context, purpose, and preferences
 */
export async function initialSchemaGeneration(
    context: string,
    purpose: string,
    preferences: string
): Promise<IdeaSchema> {
    const userPrompt = prompts.initialSchemaGeneration.userPrompt
        .replace('{{.context}}', context)
        .replace('{{.purpose}}', purpose)
        .replace('{{.preferences}}', preferences);

    const systemPrompt = prompts.initialSchemaGeneration.systemPrompt;
    
    const result = await callChatCompletion(systemPrompt, userPrompt, "gpt-4o-mini", 1);
    
    // Validate the schema structure
    if (result && typeof result === 'object' && result.purpose && result.context && result.criteria) {
        return result as IdeaSchema;
    }
    
    throw new Error('Invalid response format from schema generation');
}

/**
 * Refine schema based on user's idea rankings and preferences
 */
export async function refineSchemaBasedOnIdeaPreferences(
    schema: IdeaSchema,
    rankings: IdeaRanking[]
): Promise<IdeaSchema> {
    const userPrompt = prompts.refineSchemaBasedOnIdeaPreferences.userPrompt
        .replace('{{.schema}}', JSON.stringify(schema))
        .replace('{{.rankings}}', JSON.stringify(rankings));

    const systemPrompt = prompts.refineSchemaBasedOnIdeaPreferences.systemPrompt;
    
    const result = await callChatCompletion(systemPrompt, userPrompt, "gpt-4o-mini", 1);
    
    // Validate the schema structure
    if (result && typeof result === 'object' && result.purpose && result.context && result.criteria) {
        return result as IdeaSchema;
    }
    
    throw new Error('Invalid response format from schema refinement');
}

/**
 * Generate ideas based on the schema
 */
export async function ideaGeneration(schema: IdeaSchema): Promise<Idea[]> {
    const userPrompt = prompts.ideaGeneration.userPrompt
        .replace('{{.schema}}', JSON.stringify(schema))
        .replace('{{.purpose}}', schema.purpose);

    const systemPrompt = prompts.ideaGeneration.systemPrompt
        .replace('{{.purpose}}', schema.purpose);
    
    const result = await callChatCompletion(systemPrompt, userPrompt, "gpt-4o-mini", 1.3, true);
    
    // Ensure we return an array of ideas
    if (Array.isArray(result)) {
        return result;
    }
    
    // If result is not an array, try to extract ideas from it
    if (result && typeof result === 'object' && result.ideas) {
        return result.ideas;
    }
    
    throw new Error('Invalid response format from idea generation');
}

/**
 * Evaluate ideas based on the schema and return top 10 with evaluations
 */
export async function ideaEvaluation(
    schema: IdeaSchema,
    ideas: Idea[]
): Promise<IdeaWithEvaluation[]> {
    const userPrompt = prompts.ideaEvaluation.userPrompt
        .replace('{{.schema}}', JSON.stringify(schema))
        .replace('{{.purpose}}', schema.purpose)
        .replace('{{.ideas}}', JSON.stringify(ideas));

    const systemPrompt = prompts.ideaEvaluation.systemPrompt;
    
    const result = await callChatCompletion(systemPrompt, userPrompt, "gpt-4o-mini", 1, true);
    
    // Ensure we return an array of evaluated ideas
    if (Array.isArray(result)) {
        return result;
    }
    
    // If result is not an array, try to extract ideas from it
    if (result && typeof result === 'object' && result.ideas) {
        return result.ideas;
    }
    
    throw new Error('Invalid response format from idea evaluation');
}

// Database persistence functions

/**
 * Save a new ideation session
 */
export async function saveIdeationSession(
    context: string,
    purpose: string,
    preferences: string
): Promise<string> {
    const supabase = await createClient();

    const { data: userData, error: userError } = await supabase.auth.getUser();
    if (userError) {
        throw new Error(userError.message);
    }
    if (!userData.user) {
        throw new Error('User not found');
    }

    const { data, error } = await supabase
        .from('ideation_sessions')
        .insert({
            user_id: userData.user.id,
            context,
            purpose,
            preferences
        })
        .select('id')
        .single();

    if (error) {
        throw new Error(error.message);
    }

    return data.id;
}

/**
 * Load an ideation session
 */
export async function loadIdeationSession(sessionId: string) {
    const supabase = await createClient();

    const { data, error } = await supabase
        .from('ideation_sessions')
        .select('*')
        .eq('id', sessionId)
        .single();

    if (error) {
        throw new Error(error.message);
    }

    return data;
}

/**
 * Get all user's ideation sessions
 */
export async function getUserIdeationSessions() {
    const supabase = await createClient();

    const { data: userData, error: userError } = await supabase.auth.getUser();
    if (userError) {
        throw new Error(userError.message);
    }
    if (!userData.user) {
        throw new Error('User not found');
    }

    const { data, error } = await supabase
        .from('ideation_sessions')
        .select('*')
        .eq('user_id', userData.user.id)
        .order('created_at', { ascending: false });

    if (error) {
        throw new Error(error.message);
    }

    return data;
}

/**
 * Save a schema version
 */
export async function saveSchemaVersion(
    sessionId: string,
    schema: IdeaSchema,
    versionNumber: number
): Promise<string> {
    const supabase = await createClient();

    const { data, error } = await supabase
        .from('schema_versions')
        .insert({
            session_id: sessionId,
            version_number: versionNumber,
            schema_data: schema
        })
        .select('id')
        .single();

    if (error) {
        throw new Error(error.message);
    }

    return data.id;
}

/**
 * Get schema versions for a session
 */
export async function getSchemaVersions(sessionId: string) {
    const supabase = await createClient();

    const { data, error } = await supabase
        .from('schema_versions')
        .select('*')
        .eq('session_id', sessionId)
        .order('version_number', { ascending: true });

    if (error) {
        throw new Error(error.message);
    }

    return data;
}

/**
 * Save ideas with rankings
 */
export async function saveIdeasWithRankings(
    sessionId: string,
    schemaVersionId: string,
    ideas: { idea: string; description?: string; evaluation?: string; ranking?: number }[]
) {
    const supabase = await createClient();

    const ideasToInsert = ideas.map(idea => ({
        session_id: sessionId,
        schema_version_id: schemaVersionId,
        idea: idea.idea,
        description: idea.description || '',
        evaluation_score: idea.evaluation || '',
        user_ranking: idea.ranking || null
    }));

    const { data, error } = await supabase
        .from('ideas')
        .insert(ideasToInsert)
        .select('*');

    if (error) {
        throw new Error(error.message);
    }

    return data;
}

/**
 * Get ideas for a session
 */
export async function getSessionIdeas(sessionId: string) {
    const supabase = await createClient();

    const { data, error } = await supabase
        .from('ideas')
        .select('*')
        .eq('session_id', sessionId)
        .order('created_at', { ascending: true });

    if (error) {
        throw new Error(error.message);
    }

    return data;
}

/**
 * Update idea rankings
 */
export async function updateIdeaRankings(
    rankings: { id: string; ranking: number }[]
) {
    const supabase = await createClient();

    const updates = rankings.map(async ({ id, ranking }) => {
        const { error } = await supabase
            .from('ideas')
            .update({ user_ranking: ranking })
            .eq('id', id);

        if (error) {
            throw new Error(error.message);
        }
    });

    await Promise.all(updates);
}
